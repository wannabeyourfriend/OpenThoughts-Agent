
my tezos oracles docker image (tqtezos/oracle-stock-ticker) is unable to find activation code from faucet configuration file using flask app. Getting following error: File "/usr/local/lib/python3.6/dist-packages/pytezos/crypto.py", line 243, in from_faucet activation_code=data['secret'] KeyError: 'secret' Faucet config file do not have any secret key.
