
In order to use the verifySignature method in Taquito, I need the public key of the wallet. I can only find the public key hash method - am I missing something obvious? Thanks! Below is how I obtain the wallet public key hash: wallet.requestPermissions({ network: { type: 'hangzhounet' } }) .then(() => wallet.getPKH());
