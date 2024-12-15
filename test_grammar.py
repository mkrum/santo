from lark import Lark

# Define the grammar
grammar = open("grammar.lark", "r").read()

# Use the parser
parser = Lark(grammar, start='start', parser='lalr', debug=True)

strings = ["6", "6413", "64(1)3", "64(1)3(B)9", "4(1)3/G4/GDP", "8(B)84(2)/LDP/L8", "54(B)/BG25/SH.1-2", "1(B)16(2)63(1)/LTP/L1", "T9/F9LD.2-H", "S4/G34.2-H(E4/TH)(UR)(NR);1-3;B-2", "E6/G6.3-H(RBI);2-3;B-1", "S/L9S.3-H;2X3(5/INT);1-2", "K.1-2(WP)"]

for s in strings:
    parser.parse(s)

