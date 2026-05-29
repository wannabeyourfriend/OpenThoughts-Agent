
var crypto = require('crypto'); var ed25519 = require('./'); const seed = crypto.randomBytes(32); const keyPair = ed25519.MakeKeypair(seed); const base58encoded = base58.encode(keyPair.publicKey); console.log('base58 pub key: ' + base58encoded); When I run the above code, it shows me the following error ed25519.MakeKeypair is not a function Can anyone please guide me on how to proceed?
