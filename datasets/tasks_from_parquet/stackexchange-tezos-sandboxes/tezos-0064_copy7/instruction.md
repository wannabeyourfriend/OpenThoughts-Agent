
The node I need to connect to has a variable IP. I've set up a Dynamic DNS service and use the following workaround to connect to it: ./tezos-admin-client connect address $(host myddnsaddress.net | awk '/has address/ { print $4 ; exit }'):9732 Is there a better way to have your node connect to an URL?
