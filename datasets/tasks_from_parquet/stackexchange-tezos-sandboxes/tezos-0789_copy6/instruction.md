
I need to validate Tezos addresses and have arrived at the following prefixed base 58 regular expressions for wallet and contract addresses. Contract: /KT1[1-9A-HJ-NP-Za-km-z]{33}/ Wallet: /tz[1-3][1-9A-HJ-NP-Za-km-z]{33}/ Are these regexes sufficient, or could something like a KT2 or tz4 address ever be introduced into the protocol?
