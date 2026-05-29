
I encountered a weird behavior. If you try to send 2 consecutive tx (in the same tezos block lifetime) you get invalid counter errors: Counter [NUMBER] already used for contract [ADDRESS] Seems related to this: https://gitlab.com/tezos/tezos/issues/376 If you increase the counter you get the opposite error: contract.counter_in_the_future
