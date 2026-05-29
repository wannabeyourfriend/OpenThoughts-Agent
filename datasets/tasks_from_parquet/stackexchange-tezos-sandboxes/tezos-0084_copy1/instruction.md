
How can I derive Tezos public key from x,y points on the curve (either SECP256K1 or P-256 )? For example, for Ethereum you can compute this using Keccak-256 on the [x,y] . The address is then obtained by taking the last 40 bytes (20 hex chars) and prefixing it with 0x for a total of 42 bytes.
