
Alright, having a basic kernel working (counter example) I want to start consuming more complex (external) messages. Can I produce example message doing something like this? echo '{ "n": 2, "bar": "baz", "true": false }' | xxd -p And how should I think about adding a prefix? I’m thinking of the MAGIC_BYTE from constants in the tzwitter_app kernel example? Question from Slack.
