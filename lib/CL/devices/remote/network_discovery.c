/* network_discovery.c - part of pocl-remote driver that performs network
   discovery to find remote servers and their devices.


   Copyright (c) 2023-2024 Yashvardhan Agarwal / Tampere University

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
   deal in the Software without restriction, including without limitation the
   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
   sell copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
   IN THE SOFTWARE.
*/

#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

#include "network_discovery.h"
#include "pocl_cl.h"
#include "pocl_debug.h"
#include "pocl_runtime_config.h"
#include "pocl_threads.h"
#include "uthash.h"

#if defined(ENABLE_REMOTE_DISCOVERY_AVAHI)
#include <avahi-client/lookup.h>
#include <avahi-common/domain.h>
#include <avahi-common/error.h>
#include <avahi-common/malloc.h>
#include <avahi-common/thread-watch.h>

AvahiThreadedPoll *avahi_threaded_poll = NULL;
AvahiClient *avahi_client = NULL;
cl_int domain_count = 0;
AvahiServiceBrowser **service_browser = NULL;
#endif

#if defined(ENABLE_REMOTE_DISCOVERY_DHT)
#include <opendht/opendht_c.h>
#endif

#define FREE_INFO(__OBJ__)                                                    \
  do                                                                          \
    {                                                                         \
      POCL_MEM_FREE (__OBJ__->id);                                            \
      POCL_MEM_FREE (__OBJ__->domain);                                        \
      POCL_MEM_FREE (__OBJ__->ip_port);                                       \
    }                                                                         \
  while (0)

/**
 * This structure contains remote server inforamtion about remote server found
 * through discovery. A hash table handle (`UT_hash_handle`) is included to
 * allow instances of this struct to be managed in a hash table, preventing
 * redundant entries when the same server is discovered multiple times.
 *
 * \brief Represents information about a remote server discovered on the
 * network.
 */
typedef struct server_info_t
{
  char *id;             // Unique ID with which server advertises itself
  char *domain;         // Domain name in which the server was found
  char *ip_port;        // "IP:port"
  cl_uint device_count; // Number of devices in the remote server
  UT_hash_handle hh;
} server_info;

/* Function pointer for the callback defined in pocl runtime to dynamically
 * add and initialize devices in the discovered remote server.
 */
cl_int (*disco_dev_init) (const char *, unsigned);

/* Function pointer for the callback defined in pocl-r driver to reconnect
 * to a know server with same session.
 */
cl_int (*reconnect) (const char *);

unsigned dev_type_idx;

static server_info *server_table = NULL;
static pocl_lock_t server_info_lock = POCL_LOCK_INITIALIZER;

void
destroy_server_info_table (server_info *info)
{
  server_info *current;
  server_info *temp;

  HASH_ITER (hh, info, current, temp)
  {
    FREE_INFO (current);
    HASH_DEL (info, current);
    POCL_MEM_FREE (current);
  }
}

/**
 * Function called by the resolver to add a new server. This function calls the
 * callback that dynamically adds the devices to the platform.
 */
static void
register_server (const char *id,
                 const char *domain,
                 const char *server_key,
                 cl_uint device_count,
                 server_info *p_info)
{
  if (!device_count)
    {
      POCL_MSG_ERR ("Resolver called with zero devices.\n");
      return;
    }

  server_info *new_server = p_info;
  if (!new_server)
    {
      new_server = (server_info *)calloc (1, sizeof (*new_server));
      new_server->id = strndup (id, strlen (id));
      new_server->domain = strndup (domain, strlen (domain));
      new_server->ip_port = strndup (server_key, strlen (server_key));
    }

  new_server->device_count = device_count;

  for (int i = 0; i < device_count; i++)
    {
      char dev_param[40 + 15];
      snprintf (dev_param, sizeof (dev_param), "%s/%d", new_server->ip_port,
                i);

      cl_int err = disco_dev_init (dev_param, dev_type_idx);

      if (err)
        {
          if (!p_info)
            {
              FREE_INFO (new_server);
              POCL_MEM_FREE (new_server);
            }
          POCL_MSG_ERR ("Device couldn't be added, skipping this server.\n");
          return;
        }
    }

  if (!p_info)
    {
      HASH_ADD_KEYPTR (hh, server_table, new_server->ip_port,
                       strlen (new_server->ip_port), new_server);
    }
}

/**
 * The function takes the server information from the discovery resolvers and
 * decides to add, reconnect or ignore the passed server and its devices.
 */
static void
resolver (const char *id,
          const char *domain,
          const char *server_key,
          const char *type,
          cl_uint device_count)
{
  POCL_LOCK (server_info_lock);

  server_info *head = server_table, *p_info = NULL;

  /* address && id -> reconnect normally
   * !address && id -> ignore new address
   * address && !id -> session changed, add as new service but need to
   * handle repeatr address in find or create new server function !address &&
   * !id -> add as a new service */

  HASH_FIND_STR (head, server_key, p_info);

  if (p_info)
    {
      if (!strncmp (p_info->id, id, SERVER_ID_SIZE))
        {
          POCL_MSG_PRINT_REMOTE (
            "Avahi / DHT resolver: Server '%s' of type '%s' in "
            "domain '%s' is known with the same session.\n",
            id, type, domain);
          reconnect (server_key);
        }
      else
        {
          POCL_MSG_PRINT_REMOTE (
            "Avahi / DHT resolver: Server '%s' of type '%s' in "
            "domain '%s' is known but the old session expired. Adding "
            "Server.\n",
            id, type, domain);
          register_server (id, domain, server_key, device_count, p_info);
        }
    }
  else
    {
      for (p_info = server_table; p_info; p_info = p_info->hh.next)
        {
          if (!strncmp (p_info->id, id, SERVER_ID_SIZE))
            break;
        }

      if (p_info)
        {
          POCL_MSG_PRINT_REMOTE (
            "Avahi / DHT resolver: Server '%s' of type '%s' in "
            "domain '%s' is known and registered with a different address.\n",
            id, type, domain);
        }
      else
        {
          POCL_MSG_PRINT_REMOTE (
            "Avahi / DHT resolver: Server '%s' of type '%s' in "
            "domain '%s' is being added.\n",
            id, type, domain);
          register_server (id, domain, server_key, device_count, p_info);
        }
    }

  POCL_UNLOCK (server_info_lock);
}

/*****************************************************************************/
#if defined(ENABLE_REMOTE_DISCOVERY_AVAHI)

/**
 *  Called to stop and clear avahi.
 */
static void
clear_avahi ()
{
  if (service_browser)
    {
      for (int i = 0; i < domain_count; i++)
        avahi_service_browser_free (service_browser[i]);

      free (service_browser);
      service_browser = NULL;
    }

  if (avahi_client)
    {
      avahi_client_free (avahi_client);
      avahi_client = NULL;
    }

  if (avahi_threaded_poll)
    {
      avahi_threaded_poll_free (avahi_threaded_poll);
      avahi_threaded_poll = NULL;
    }
}

/**
 * To be called by the resolver when a new service of type "_pocl._tcp" is
 * found and has to be resolved.
 */
static void
avahi_resolve_callback (AvahiServiceResolver *r,
                        AVAHI_GCC_UNUSED AvahiIfIndex interface,
                        AVAHI_GCC_UNUSED AvahiProtocol protocol,
                        AvahiResolverEvent event,
                        const char *name,
                        const char *type,
                        const char *domain,
                        const char *host_name,
                        const AvahiAddress *address,
                        uint16_t port,
                        AvahiStringList *txt,
                        AvahiLookupResultFlags flags,
                        AVAHI_GCC_UNUSED void *userdata)
{
  assert (r);

  char addr[AVAHI_ADDRESS_STR_MAX];
  char server_key[AVAHI_ADDRESS_STR_MAX + 15];

  switch (event)
    {
    case AVAHI_RESOLVER_FOUND:

      avahi_address_snprint (addr, sizeof (addr), address);
      snprintf (server_key, sizeof (server_key), "%s:%d", addr, port);

      char *text = avahi_string_list_to_string (txt);
      /* avahi_string_list_to_string return text field enclosed in "" */
      cl_uint dev_count = strlen (text) - 2;
      avahi_free (text);

      resolver (name, domain, server_key, type, dev_count);

      break;

    case AVAHI_RESOLVER_FAILURE:
      POCL_MSG_ERR (
        "Avahi resolver failed to resolve '%s' of type '%s' in domain "
        "'%s': %s\n",
        name, type, domain,
        avahi_strerror (avahi_client_errno (avahi_client)));
      break;
    }

  avahi_service_resolver_free (r);
}

/**
 * To be called by the browser when a new service of type "_pocl._tcp" is
 * found or a service is removed.
 */
static void
avahi_browser_callback (AvahiServiceBrowser *b,
                        AvahiIfIndex interface,
                        AvahiProtocol protocol,
                        AvahiBrowserEvent event,
                        const char *name,
                        const char *type,
                        const char *domain,
                        AVAHI_GCC_UNUSED AvahiLookupResultFlags flags,
                        void *userdata)
{
  assert (b);

  switch (event)
    {
    case AVAHI_BROWSER_NEW:
      POCL_MSG_PRINT_REMOTE (
        "Avahi Browser: FOUND new server '%s', type '%s' in domain '%s'.\n",
        name, type, domain);
      if (!(avahi_service_resolver_new (avahi_client, interface, protocol,
                                        name, type, domain, AVAHI_PROTO_UNSPEC,
                                        0, avahi_resolve_callback, NULL)))
        POCL_MSG_ERR (
          "Avahi browser callback failed to call resolver '%s': %s\n", name,
          avahi_strerror (avahi_client_errno (avahi_client)));
      break;

    case AVAHI_BROWSER_REMOVE:
      POCL_MSG_PRINT_REMOTE (
        "Avahi Browser: LOST server '%s', type '%s' in domain '%s'.\n", name,
        type, domain);
      break;

    case AVAHI_BROWSER_FAILURE:
      POCL_MSG_ERR ("Avahi Browser Callback: %s \n",
                    avahi_strerror (avahi_client_errno (avahi_client)));
      break;

    case AVAHI_BROWSER_ALL_FOR_NOW:
      break;

    case AVAHI_BROWSER_CACHE_EXHAUSTED:
      break;
    }
}

/**
 * Called when avahi client or server state changes.
 */
static void
avahi_client_callback (AvahiClient *c,
                       AvahiClientState state,
                       AVAHI_GCC_UNUSED void *userdata)
{
  assert (c);

  if (state == AVAHI_CLIENT_FAILURE)
    {
      POCL_MSG_ERR ("Avahi client connection failure: %s\n",
                    avahi_strerror (avahi_client_errno (c)));
      clear_avahi ();
    }
}

/**
 * Initialize Avahi thread and listener to look for remote servers in local
 * and other specified domains.
 */
cl_int
init_avahi_discovery ()
{
  cl_int errcode = CL_SUCCESS;
  cl_int avahi_errcode;

  /* Avahi polling thread */
  avahi_threaded_poll = avahi_threaded_poll_new ();
  POCL_GOTO_ERROR_ON (!avahi_threaded_poll, CL_OUT_OF_RESOURCES,
                      "Avahi failed to create threaded poll object.\n");
  /* Avahi client */
  avahi_client
    = avahi_client_new (avahi_threaded_poll_get (avahi_threaded_poll), 0,
                        avahi_client_callback, NULL, &avahi_errcode);
  POCL_GOTO_ERROR_ON (!(avahi_client), CL_OUT_OF_RESOURCES,
                      "Avahi failed to create client object with error: %s\n",
                      avahi_strerror (avahi_errcode));

  /* Get the search domains from the environment. These domains including
   * .local are used to search for available remote servers that are being
   * advertised through m-DNS or are registered in name servers of the
   * specified domain(s). */
  const char *env = pocl_get_string_option (POCL_REMOTE_SEARCH_DOMAINS, NULL);
  if (env && *env)
    {
      /* Count the number of domains in which discovery should be conducted */
      char *domains, *token, *saveptr = NULL;
      domains = strdup (env);
      /* .local + at least one other domain in env as env is not null */
      domain_count = 1 + 1;
      token = strtok_r (domains, ",", &saveptr);
      while (token)
        {
          token = strtok_r (NULL, ",", &saveptr);
          domain_count++;
        }
      token = NULL;
      saveptr = NULL;
      POCL_MEM_FREE (domains);

      /* Allocate memory for Avahi browser for each domain */
      service_browser = (AvahiServiceBrowser **)malloc (
        domain_count * sizeof (AvahiServiceBrowser *));

      /* Initialize browser for .local domain */
      service_browser[0] = avahi_service_browser_new (
        avahi_client, AVAHI_IF_UNSPEC, AVAHI_PROTO_UNSPEC,
        POCL_REMOTE_DNS_SRV_TYPE, NULL, 0, avahi_browser_callback, NULL);
      POCL_GOTO_ERROR_ON (
        !service_browser[0], CL_OUT_OF_RESOURCES,
        "Avahi failed to create service browser with error: %s\n",
        avahi_strerror (avahi_client_errno (avahi_client)));

      /* Loop to setup browsers in the env specified domains */
      domains = strdup (env);
      token = strtok_r (domains, ",", &saveptr);
      while (token)
        {
          service_browser[0] = avahi_service_browser_new (
            avahi_client, AVAHI_IF_UNSPEC, AVAHI_PROTO_UNSPEC, token, NULL, 0,
            avahi_browser_callback, NULL);
          POCL_GOTO_ERROR_ON (
            !service_browser[0], CL_OUT_OF_RESOURCES,
            "Avahi failed to create service browser with error: %s\n",
            avahi_strerror (avahi_client_errno (avahi_client)));

          token = strtok_r (NULL, ",", &saveptr);
        }

      POCL_MEM_FREE (domains);
    }
  else
    {
      /* Browser only for .local if env for domains is null */
      domain_count = 1;
      service_browser
        = (AvahiServiceBrowser **)malloc (sizeof (AvahiServiceBrowser *));

      service_browser[0] = avahi_service_browser_new (
        avahi_client, AVAHI_IF_UNSPEC, AVAHI_PROTO_UNSPEC,
        POCL_REMOTE_DNS_SRV_TYPE, NULL, 0, avahi_browser_callback, NULL);
      POCL_GOTO_ERROR_ON (
        !service_browser[0], CL_OUT_OF_RESOURCES,
        "Avahi failed to create service browser with error: %s\n",
        avahi_strerror (avahi_client_errno (avahi_client)));
    }

  /* Start the avahi thread*/
  avahi_errcode = avahi_threaded_poll_start (avahi_threaded_poll);
  POCL_GOTO_ERROR_ON ((avahi_errcode < 0), CL_OUT_OF_RESOURCES,
                      "Avahi failed to start threaded poll with error: %s\n",
                      avahi_strerror (avahi_errcode));
  POCL_MSG_PRINT_REMOTE ("Avahi Browser: Browsing started \n");

  return errcode;

ERROR:

  clear_avahi ();
  return errcode;
}
#endif

/*****************************************************************************/
#if defined(ENABLE_REMOTE_DISCOVERY_DHT)

const char *common_key = NULL;

struct op_context
{
  dht_runner *node;
  atomic_bool stop;
};

struct listen_context
{
  dht_runner *node;
  dht_op_token *token;
  size_t count;
};

bool
dht_value_callback (const dht_value *value, bool expired, void *user_data)
{

  struct listen_context *ctx = (struct listen_context *)user_data;

  if (expired)
    ctx->count--;
  else
    ctx->count++;

  dht_data_view data = dht_value_get_data (value);
  char
    key[55]; /* size of AVAHI_ADDRESS_STR_MAX + 15, which is max size of IP */
             /* address + 15 */
  char id[33]; /* server id is 32char + '\0' */
  int dev_count = 0;

  /* The device information from the server is separated using ~ as the
   * delimiter. The client should know how to interpret/decode the server
   * data. */
  char *ddata;
  char *token;
  ddata = strndup ((char *)data.data, data.size);

  token = strtok (ddata, "~");
  snprintf (id, strlen (token) + 1, "%s", token);

  token = strtok (NULL, "~");
  snprintf (key, strlen (token) + 1, "%s", token);

  while (token)
    {
      token = strtok (NULL, "~");
      dev_count++;
    }
  POCL_MEM_FREE (ddata);
  dev_count = dev_count / 4;

  resolver (id, common_key, key, "DHT", dev_count);

  return true;
}

void
listen_context_free (void *user_data)
{
  struct listen_context *ctx = (struct listen_context *)user_data;
  dht_op_token_delete (ctx->token);
  POCL_MEM_FREE (ctx);
}

void
dht_shutdown_callback (void *user_data)
{
  struct op_context *ctx = (struct op_context *)user_data;
  atomic_store (&ctx->stop, true);
}

dht_infohash
get_hash (const char *key_str)
{
  dht_infohash key;
  dht_infohash_from_hex_null (&key, key_str);
  if (dht_infohash_is_zero (&key))
    {
      dht_infohash_get_from_string (&key, key_str);
    }
  return key;
}

/**
 * Initialize DHT node and start the listener on the DHT network and the given
 * key.
 */
cl_int
init_dht_discovery ()
{
  cl_int errcode = CL_SUCCESS;

  dht_runner *node = dht_runner_new ();
  dht_runner_config dht_config;
  dht_runner_config_default (&dht_config);

  /* Port to start the DHT taken from environment or default used.*/
  const int port = pocl_get_int_option (POCL_REMOTE_DHT_PORT, 4222);
  /* DHT bootstrap node taken from environment or default used.*/
  const char *bootstrap
    = pocl_get_string_option (POCL_REMOTE_DHT_BOOTSTRAP, "bootstrap.jami.net");
  /* Common key used by the servers and clients participating in the DHT
   * network to find or publish remote server information. Taken from
   * environment or default used. */
  common_key
    = pocl_get_string_option (POCL_REMOTE_DHT_KEY, "poclremoteservernetwork");

  dht_runner_run_config (node, port, &dht_config);
  dht_runner_bootstrap (node, bootstrap, NULL);

  dht_infohash hash_id;

  hash_id = get_hash (common_key);

  struct listen_context *ctx = malloc (sizeof (struct listen_context));
  ctx->node = node;
  ctx->count = 0;
  ctx->token = dht_runner_listen (node, &hash_id, dht_value_callback,
                                  listen_context_free, ctx);
  POCL_MSG_PRINT_REMOTE ("DHT node %s running on port %u\n",
                         dht_infohash_print (&hash_id),
                         dht_runner_get_bound_port (node, AF_INET));

  return errcode;
}
#endif

/*****************************************************************************/

/**
 * Function called by the remote driver to initialize network disocvery
 * methods.
 */
cl_int
init_network_discovery (cl_int (*disco_dev_init_callback) (const char *,
                                                           unsigned),
                        cl_int (*reconnect_callback) (const char *),
                        unsigned pocl_dev_type_idx)
{
  cl_int errcode = CL_SUCCESS;
  disco_dev_init = disco_dev_init_callback;
  reconnect = reconnect_callback;
  dev_type_idx = pocl_dev_type_idx;

#if defined(ENABLE_REMOTE_DISCOVERY_AVAHI)
  errcode = init_avahi_discovery ();
  if (errcode != CL_SUCCESS)
    return errcode;
#endif

#if defined(ENABLE_REMOTE_DISCOVERY_DHT)
  errcode = init_dht_discovery ();
  if (errcode != CL_SUCCESS)
    return errcode;
#endif
  return errcode;
}