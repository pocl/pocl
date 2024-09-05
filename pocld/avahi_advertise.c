/* avahi_advertise.c - part of pocl-daemon that performs mDNS on local network
 to advertise the remote server and its devices.


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

#include <assert.h>
#include <string.h>

#include "avahi_advertise.h"
#include "pocl_debug.h"

#include <avahi-client/publish.h>
#include <avahi-common/alternative.h>
#include <avahi-common/error.h>
#include <avahi-common/thread-watch.h>

typedef struct server_info_t
{
  char *id;        /* Unique ID with which server advertises itself */
  cl_int if_index; /* Interface index to use to advertise the server */
  cl_int ip_proto; /* IPv4 or IPv6 or both */
  uint16_t port;   /* Port used by the server */
  char *info;      /* Server's device info */
} server_info;

server_info server;
AvahiThreadedPoll *avahi_threaded_poll = NULL;
AvahiClient *avahi_client = NULL;
AvahiEntryGroup *avahi_entry_group = NULL;
static void create_service (AvahiClient *c);

static void
clear_avahi ()
{

  /* clean up*/
  if (avahi_entry_group)
    avahi_entry_group_free (avahi_entry_group);
  if (avahi_client)
    avahi_client_free (avahi_client);
  if (avahi_threaded_poll)
    avahi_threaded_poll_free (avahi_threaded_poll);

  free (server.id);
  free (server.info);
}

/**
 * Called whenever the entry group's state changes.
 */
static void
entry_group_callback (AvahiEntryGroup *g,
                      AvahiEntryGroupState state,
                      AVAHI_GCC_UNUSED void *userdata)
{
  assert (g == avahi_entry_group || avahi_entry_group == NULL);
  avahi_entry_group = g;

  switch (state)
    {
    case AVAHI_ENTRY_GROUP_ESTABLISHED:
      POCL_MSG_PRINT_REMOTE ("Avahi service '%s' successfully established.\n",
                             server.id);
      break;

    case AVAHI_ENTRY_GROUP_COLLISION:
      {
        /* Service name collision happened with another remote service, so the
         * name is modified. */
        char *alt;
        alt = avahi_alternative_service_name (server.id);
        free (server.id);
        server.id = alt;
        POCL_MSG_PRINT_REMOTE (
          "Avahi service name collision, renaming to '%s'\n", server.id);
        /* Recreate the service with the new name. */
        create_service (avahi_entry_group_get_client (g));
        break;
      }

    case AVAHI_ENTRY_GROUP_FAILURE:
      POCL_MSG_ERR ("Avahi entry group failure: %s.\n",
                    avahi_strerror (
                      avahi_client_errno (avahi_entry_group_get_client (g))));
      clear_avahi ();
      break;

    case AVAHI_ENTRY_GROUP_UNCOMMITED:
    case AVAHI_ENTRY_GROUP_REGISTERING:
      break;
    }
}

/**
 * Adds the server as a service advertised using mDNS by the host.
 */
static void
create_service (AvahiClient *c)
{
  assert (c);

  /* Create new entry group when the callback is being called for the first
   * time. */
  if (!avahi_entry_group)
    {
      avahi_entry_group
        = avahi_entry_group_new (c, entry_group_callback, NULL);
      if (!avahi_entry_group)
        {
          POCL_MSG_ERR ("Avahi failed to create new entry group.\n");
          goto ERROR;
        }
    }

  cl_int err;
  /* If the entry group is empty then we add a service that advertises the
   * pocl-r server. */
  if (avahi_entry_group_is_empty (avahi_entry_group))
    {
      POCL_MSG_PRINT_REMOTE (
        "Avahi adding the remote server as service '%s'\n", server.id);

      err = avahi_entry_group_add_service (
        avahi_entry_group, server.if_index, server.ip_proto, 0, server.id,
        "_pocl._tcp", NULL, NULL, server.port, server.info, NULL);
      if (err < 0)
        {
          /* Service name collision happened with another local service, so the
           * name is modified. */
          if (err == AVAHI_ERR_COLLISION)
            {
              char *alt = NULL;
              alt = avahi_alternative_service_name (server.id);
              free (server.id);
              server.id = alt;
              POCL_MSG_PRINT_REMOTE (
                "Avahi service name collision, renaming to '%s'\n", server.id);
              /* Recreate the service with the new name. */
              create_service (c);
            }
          POCL_MSG_ERR ("Avahi failed to add _pocl._tcp service: %s\n",
                        avahi_strerror (err));
          goto ERROR;
        }

      /* Register the service */
      err = avahi_entry_group_commit (avahi_entry_group);
      if (err < 0)
        {
          POCL_MSG_ERR (
            "Avahi failed to commit _pocl._tcp service to entry gorup. \n");
          goto ERROR;
        }
    }
  return;
ERROR:
  clear_avahi ();
}

/**
 * Called when avahi client state changes.
 */
static void
avahi_client_callback (AvahiClient *c,
                       AvahiClientState state,
                       AVAHI_GCC_UNUSED void *userdata)
{
  assert (c);
  switch (state)
    {
    case AVAHI_CLIENT_S_RUNNING:
      /* At this point avahi client is establihed and has registered its host
       * name on the network. Now the mDNS service can be created. */
      create_service (c);
      break;

    case AVAHI_CLIENT_FAILURE:
      POCL_MSG_ERR ("Failed to establish Avahi client with error: %s\n",
                    avahi_strerror (avahi_client_errno (c)));
      clear_avahi ();
      break;

    case AVAHI_CLIENT_S_COLLISION:

    case AVAHI_CLIENT_S_REGISTERING:
      if (avahi_entry_group)
        avahi_entry_group_reset (avahi_entry_group);
      break;
    case AVAHI_CLIENT_CONNECTING:
      break;
    }
}

/**
 * Called by the daemon to start advertising using Avahi.
 */
void
init_avahi_advertisement (const char *server_id,
                          size_t server_id_len,
                          cl_int if_index,
                          cl_int ip_proto,
                          uint16_t port,
                          const char *info,
                          size_t info_len)
{

  cl_int avahi_errcode;

  server.id = strndup (server_id, server_id_len);
  server.if_index = if_index;
  server.ip_proto = ip_proto;
  server.port = port;
  server.info = strndup (info, info_len);

  /* Avahi polling thread */
  avahi_threaded_poll = avahi_threaded_poll_new ();
  if (!avahi_threaded_poll)
    {
      POCL_MSG_ERR ("Avahi failed to create threaded poll object.\n");
      goto ERROR;
    }

  /* Avahi client */
  avahi_client
    = avahi_client_new (avahi_threaded_poll_get (avahi_threaded_poll), 0,
                        avahi_client_callback, NULL, &avahi_errcode);
  if (!avahi_client)
    {
      POCL_MSG_ERR ("Avahi failed to create client object with error: %s\n",
                    avahi_strerror (avahi_errcode));
      goto ERROR;
    }

  /* Start the avahi thread */
  avahi_errcode = avahi_threaded_poll_start (avahi_threaded_poll);
  if (avahi_errcode < 0)
    {
      POCL_MSG_ERR ("Avahi failed to start threaded poll with error: %s\n",
                    avahi_strerror (avahi_errcode));
      goto ERROR;
    }

  return;

ERROR:
  clear_avahi ();
}
