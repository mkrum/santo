
import santo.data

def test_evn():
    games = santo.data.load_evn("./data/1992/1992CHN.EVN")

def test_eva():
    games = santo.data.load_evn("./data/1992/1992MIL.EVA")

def test_box_score():
    games = santo.data.load_evn("./data/1992/1992CHN.EVN")
    games[0].box_score()
