
I would like to compile all contracts in a folder at one go. I tried "compile all": "docker run --rm -v \"$PWD\":\"$PWD\" -w \"$PWD\" ligolang/ligo:0.24.0 compile-contract contracts/*.ligo main > compiled/*.tz",
