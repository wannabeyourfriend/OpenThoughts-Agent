
When injecting a transaction from a tz1 account to another one, tezos-client will add 100 to the estimated gas. We know that such operation consumes 1427 gas, why increasing it by default? What could go wrong otherwise?
