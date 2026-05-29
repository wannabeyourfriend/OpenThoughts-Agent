
When querying a block via RPC, the balance_updates array found inside the metadata object may have entries with negative rewards. For instance: { "kind":"freezer", "category":"rewards", "delegate":"tz2KuCcKSyMzs8wRJXzjqoHgojPkSUem8ZBS", "level":29, "change":"-542000000" } So, what does it mean when a delegate get a negative reward?
