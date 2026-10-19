def test_package_importable():
    import movietaggen
    assert movietaggen.__file__ is not None
