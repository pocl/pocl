/* avahi_advertise.cc - part of pocl-daemon that performs mDNS on local network
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

#include <cassert>

#include "avahi_advertise.hh"
#include "pocl_debug.h"

#include <avahi-client/client.h>
#include <avahi-client/publish.h>
#include <avahi-common/alternative.h>
#include <avahi-common/error.h>
#include <avahi-common/malloc.h>
#include <avahi-common/thread-watch.h>

// Called whenever the entry group's state changes.
void AvahiAdvertise::entryGroupCallback(AvahiEntryGroup *g,
                                        AvahiEntryGroupState state,
                                        AVAHI_GCC_UNUSED void *userdata) {
  AvahiAdvertise *AA = static_cast<AvahiAdvertise *>(userdata);
  assert(g == AA->avahiEntryGroup || AA->avahiEntryGroup == NULL);
  AA->avahiEntryGroup = g;

  switch (state) {
  case AVAHI_ENTRY_GROUP_ESTABLISHED:
    POCL_MSG_PRINT_REMOTE("Avahi service '%s' successfully established.\n",
                          AA->server.id.c_str());
    break;

  case AVAHI_ENTRY_GROUP_COLLISION: {
    // Service name collision happened with another remote service, so the
    // name is modified.
    char *alt;
    alt = avahi_alternative_service_name(AA->server.id.c_str());
    AA->server.id = alt;
    avahi_free(alt);
    POCL_MSG_PRINT_REMOTE("Avahi service name collision, renaming to '%s'\n",
                          AA->server.id.c_str());
    // Recreate the service with the new name.
    createService(avahi_entry_group_get_client(g), userdata);
    break;
  }

  case AVAHI_ENTRY_GROUP_FAILURE:
    POCL_MSG_ERR(
        "Avahi entry group failure: %s.\n",
        avahi_strerror(avahi_client_errno(avahi_entry_group_get_client(g))));
    AA->clearAvahi();
    break;

  case AVAHI_ENTRY_GROUP_UNCOMMITED:
  case AVAHI_ENTRY_GROUP_REGISTERING:
    break;
  }
}

// Adds the server as a service advertised using mDNS by the host.
void AvahiAdvertise::createService(AvahiClient *c, void *userdata) {
  assert(c);
  AvahiAdvertise *AA = static_cast<AvahiAdvertise *>(userdata);
  // Create new entry group when the callback is being called for the first
  // time.
  if (AA->avahiEntryGroup == NULL) {
    AA->avahiEntryGroup = avahi_entry_group_new(c, entryGroupCallback, AA);
    if (AA->avahiEntryGroup == NULL) {
      POCL_MSG_ERR("Avahi failed to create new entry group.\n");
      goto ERROR;
    }
  }

  cl_int err;
  // If the entry group is empty then we add a service that advertises the
  // pocl-r server.
  if (avahi_entry_group_is_empty(AA->avahiEntryGroup)) {
    POCL_MSG_PRINT_REMOTE("Avahi adding the remote server as service '%s'\n",
                          AA->server.id.c_str());

    err = avahi_entry_group_add_service(
        AA->avahiEntryGroup, AA->server.ifIndex, AA->server.ipProto,
        static_cast<AvahiPublishFlags>(0), AA->server.id.c_str(), "_pocl._tcp",
        NULL, NULL, AA->server.port, AA->server.info.c_str(), NULL);
    if (err < 0) {
      // Service name collision happened with another local service, so the
      // name is modified.
      if (err == AVAHI_ERR_COLLISION) {
        char *alt = NULL;
        alt = avahi_alternative_service_name(AA->server.id.c_str());
        AA->server.id = alt;
        avahi_free(alt);
        POCL_MSG_PRINT_REMOTE(
            "Avahi service name collision, renaming to '%s'\n",
            AA->server.id.c_str());
        // Recreate the service with the new name.
        createService(c, userdata);
      }
      POCL_MSG_ERR("Avahi failed to add _pocl._tcp service: %s\n",
                   avahi_strerror(err));
      goto ERROR;
    }

    // Register the service
    err = avahi_entry_group_commit(AA->avahiEntryGroup);
    if (err < 0) {
      POCL_MSG_ERR(
          "Avahi failed to commit _pocl._tcp service to entry gorup. \n");
      goto ERROR;
    }
  }
  return;
ERROR:
  AA->clearAvahi();
}

// Called when avahi client state changes.
void AvahiAdvertise::avahiClientCallback(AvahiClient *c, AvahiClientState state,
                                         AVAHI_GCC_UNUSED void *userdata) {
  assert(c);
  AvahiAdvertise *AA = static_cast<AvahiAdvertise *>(userdata);
  switch (state) {
  case AVAHI_CLIENT_S_RUNNING:
    // At this point avahi client is establihed and has registered its host
    // name on the network. Now the mDNS service can be created.
    createService(c, AA);
    break;

  case AVAHI_CLIENT_FAILURE:
    POCL_MSG_ERR("Failed to establish Avahi client with error: %s\n",
                 avahi_strerror(avahi_client_errno(c)));
    AA->clearAvahi();
    break;

  case AVAHI_CLIENT_S_COLLISION:

  case AVAHI_CLIENT_S_REGISTERING:
    if (AA->avahiEntryGroup)
      avahi_entry_group_reset(AA->avahiEntryGroup);
    break;
  case AVAHI_CLIENT_CONNECTING:
    break;
  }
}

void AvahiAdvertise::clearAvahi() {
  // clean up
  if (avahiEntryGroup)
    avahi_entry_group_free(avahiEntryGroup);
  if (avahiClient)
    avahi_client_free(avahiClient);
  if (avahiThreadedPoll)
    avahi_threaded_poll_free(avahiThreadedPoll);
}

AvahiAdvertise::~AvahiAdvertise() { clearAvahi(); }

// Called by the daemon to start advertising using Avahi.
void AvahiAdvertise::launchAvahiAdvertisement(std::string serverID,
                                              cl_int ifIndex, cl_int ipProto,
                                              uint16_t port, std::string info) {

  cl_int avahi_errcode;

  server.id = serverID;
  server.ifIndex = ifIndex;
  server.ipProto = ipProto;
  server.port = port;
  server.info = info;

  // Avahi polling thread
  avahiThreadedPoll = avahi_threaded_poll_new();
  if (!avahiThreadedPoll) {
    POCL_MSG_ERR("Avahi failed to create threaded poll object.\n");
    goto ERROR;
  }

  // Avahi client
  avahiClient = avahi_client_new(avahi_threaded_poll_get(avahiThreadedPoll),
                                 static_cast<AvahiClientFlags>(0),
                                 avahiClientCallback, this, &avahi_errcode);
  if (!avahiClient) {
    POCL_MSG_ERR("Avahi failed to create client object with error: %s\n",
                 avahi_strerror(avahi_errcode));
    goto ERROR;
  }

  // Start the avahi thread
  avahi_errcode = avahi_threaded_poll_start(avahiThreadedPoll);
  if (avahi_errcode < 0) {
    POCL_MSG_ERR("Avahi failed to start threaded poll with error: %s\n",
                 avahi_strerror(avahi_errcode));
    goto ERROR;
  }

  return;

ERROR:
  clearAvahi();
}