
I try to get smart-contract storage data: eztz.node.setProvider('https://alphanet-node.tzscan.io') eztz.contract.watch(addr, 2, function(s){ console.log("New storage", s); }); But got an error: TypeError: contract.storage is not a function Also I tried to find API method for this on tzscan. Any idea for receive storage data? Thanks in advance for your help
