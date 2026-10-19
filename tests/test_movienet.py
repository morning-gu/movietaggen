"""MovieNet 适配器：假定格式解析（合成样本即该格式的镜像）。"""
from movietaggen.data.movienet import MovieMultimodal, load_movienet, load_srt
from movietaggen.data.sample import make_synthetic_dataset


def test_load_movienet_from_synthetic(tmp_path):
    make_synthetic_dataset(tmp_path)
    movies = load_movienet(tmp_path / "raw" / "movienet")
    assert len(movies) == 5
    m = next(x for x in movies if x.movienet_id == "mn01")
    assert m.title == "Crimson Horizon"
    assert m.year == 2018
    assert m.imdb_id == "tt0000001"
    assert len(m.keyframes) == 8
    assert all(p.suffix == ".jpg" for p in m.keyframes)
    assert m.script is not None and "CRIMSON HORIZON" in m.script
    assert m.subtitles is not None and "change everything" in m.subtitles


def test_load_movienet_missing_optional_files(tmp_path):
    make_synthetic_dataset(tmp_path)
    # 删除 mn01 的剧本 → script=None 但不崩溃
    (tmp_path / "raw" / "movienet" / "scripts" / "mn01.txt").unlink()
    movies = load_movienet(tmp_path / "raw" / "movienet")
    m = next(x for x in movies if x.movienet_id == "mn01")
    assert m.script is None
    assert m.subtitles is not None


def test_load_srt(tmp_path):
    srt = tmp_path / "a.srt"
    srt.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nHello\n\n"
        "2\n00:00:03,000 --> 00:00:04,000\nWorld\n", encoding="utf-8")
    assert load_srt(srt) == "Hello\nWorld"
