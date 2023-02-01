import rsa
from sys import argv

privkey_string = '''-----BEGIN RSA PRIVATE KEY-----
MIIBPQIBAAJBAM5vWtXtFHIRNaxKBuc77+KNtbcj/IdHnonKE95jgYWkYItVGtQ5
Mwh0RKal51+vKR2d7hdBTKbGL34YFVgryq0CAwEAAQJAfozlMp/UGNlT/fqjoL2t
oUUeUNdOx9+v4OhwpbR57PdKJt4A0ZnEL9NtPKXBsHz6ukHY6tsvYI8emmPz+zUK
AQIjAPO96+uX3iactmTyJdw1lkjk4XGJoJsI03LIff7M1VG+PfkCHwDY0R84O1Qk
PeKvISHPNEFbpgAEg9VJpwmgL53Er1UCIwDJzvgk6mse4SYLUoqSVzQFSqx2ixMV
Ciu4n9PiQcplkfRRAh8AsmhJnzS6hOPjbqX9swlVqntK0mxEikmHkyb7VEfNAiJP
JtFKFC/QbOHsoThbZMYx5p5ny8DyBdmb7Gre08NL3gdw
-----END RSA PRIVATE KEY-----'''

if __name__ == "__main__":
    privkey = rsa.PrivateKey.load_pkcs1(privkey_string)
    signature = rsa.sign(('-' + argv[1] + '-' + argv[2]).encode(), privkey, 'SHA-1')
    with open("activation_code.txt", 'w') as f:
        f.write(signature.hex())
        f.write('-'+argv[1])
        f.write('-'+argv[2])