/* dht_advertise.cc - part of pocl-daemon that establishes DHT based discovery
 service to advertise the remote server and its devices.


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

#include "dht_advertise.hh"
#include "pocl_debug.h"
#include "pocl_runtime_config.h"
#include <opendht.h>

void SetConfig(dht::DhtRunner::Config *cfg) {
  dht::crypto::Identity id{};

  auto node_ca = std::make_unique<dht::crypto::Identity>(
      dht::crypto::generateEcIdentity("DHT Node CA"));
  id = dht::crypto::generateIdentity("DHT Node", *node_ca);

  cfg->dht_config.node_config.network = 0;
  cfg->dht_config.node_config.maintain_storage = false;
  cfg->dht_config.node_config.persist_path = {};
  cfg->dht_config.node_config.public_stable = false;
  cfg->dht_config.id = id;
  cfg->dht_config.cert_cache_all = static_cast<bool>(id.first);
  cfg->threaded = true;
  cfg->proxy_server = {};
  cfg->push_node_id = "dhtnode";
  cfg->push_token = {};
  cfg->peer_discovery = false;
  cfg->peer_publish = false;
  cfg->dht_config.node_config.max_req_per_sec = -1;
  cfg->dht_config.node_config.max_peer_req_per_sec = -1;
  cfg->dht_config.node_config.max_searches = -1;
  cfg->dht_config.node_config.max_store_size = -1;
}

/// Called by the daemon to start advertising using DHT using a seprate thread.
///
/// \param info contains the serialized data that is to be published on the DHT
/// network. Data fields shared through info should be known to the client.

void initDHTAdvertisement(std::vector<uint8_t> info) {

  dht::DhtRunner node;
  dht::DhtRunner::Config cfg{};
  SetConfig(&cfg);

  // Port to start the DHT taken from environment or default used.
  const int port = pocl_get_int_option(POCL_REMOTE_DHT_PORT_ENV, 4222);
  // DHT bootstrap node taken from environment or default used.
  const char *bootstrap =
      pocl_get_string_option(POCL_REMOTE_DHT_BOOTSTRAP_ENV, NULL);
  if (bootstrap == NULL) {
    POCL_MSG_ERR("DHT Bootstrap node environment variable not specified, "
                 "server advertisement failed! \n");
    return;
  }
  // Common key used by the servers and clients participating in the DHT network
  // to find or publish remote server information. Taken from environment or
  // default used.
  const char *common_key = pocl_get_string_option(POCL_REMOTE_DHT_KEY_ENV,
                                                  "poclremoteservernetwork");

  // Start the DHT node
  node.run(port, cfg);
  node.bootstrap(bootstrap, "4222");

  POCL_MSG_PRINT_REMOTE("Starting DHT node. \n");
  dht::InfoHash hash;
  hash = dht::InfoHash(common_key);
  if (!hash) {
    hash = dht::InfoHash::get(common_key);
  }

  // The thread sleeps for 10mins after publishing the server details on the DHT
  // network. The published info remains on the network for 10mins, this is the
  // default configuration of the DHT netowork. Currently we use the public DHT
  // network: bootstrap.jami.net where the info remains for 10mins. In case a
  // new network is established with a different time then the sleep time should
  // be changed accordingly.
  const auto timeWindow = std::chrono::seconds(600);
  POCL_MSG_PRINT_REMOTE("Publishing on DHT. \n");
  while (true) {

    node.putSigned(hash, dht::Value(info.data(), info.size()), [](bool ok) {
      if (not ok) {
        POCL_MSG_ERR("Message publishing failed ! \n");
      }
    });
    std::this_thread::sleep_for(timeWindow);
  }

  node.join();
}