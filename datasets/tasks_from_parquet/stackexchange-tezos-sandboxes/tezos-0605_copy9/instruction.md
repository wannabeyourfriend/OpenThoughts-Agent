
const op = await contract.methods .buy(contractParams?.height, contractParams?.width) .send(); I have a buy entrypoint and would like to pass in some parameters and some tez at the same time. How should this be done?
