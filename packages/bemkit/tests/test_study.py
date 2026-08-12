"""load_study: model.py [+ params.toml] -> {variant name: Model}.

A study is Python, so these tests deliberately don't care what build()/
variants() return -- load_study never inspects the value, so plain sentinels
(strings, dicts) stand in for a real Model and keep the tests decoupled from
bemkit.model.
"""

import pytest

from bemkit.study import load_study


def _write(path, name, text):
    (path / name).write_text(text)


# -- build() only ------------------------------------------------------------


def test_build_only_gives_one_default_variant(tmp_path):
    _write(tmp_path, "model.py", "def build():\n    return 'the-model'\n")
    result = load_study(tmp_path)
    assert result == {"default": "the-model"}


def test_build_only_ignores_a_stray_params_toml(tmp_path):
    # build() takes no argument, so even a params.toml sitting next to it
    # must never be forced onto it.
    _write(tmp_path, "model.py",
           "def build():\n    return 'unparametrised'\n")
    _write(tmp_path, "params.toml", "mesh = 'coarse'\n")
    result = load_study(tmp_path)
    assert result == {"default": "unparametrised"}


# -- build(params) ------------------------------------------------------------


def test_params_toml_reaches_build_params(tmp_path):
    _write(tmp_path, "model.py",
           "def build(params):\n    return params['mesh']\n")
    _write(tmp_path, "params.toml", "mesh = 'fine'\n")
    result = load_study(tmp_path)
    assert result == {"default": "fine"}


def test_build_params_with_no_params_toml_gets_empty_dict(tmp_path):
    _write(tmp_path, "model.py",
           "def build(params):\n    return dict(params)\n")
    result = load_study(tmp_path)
    assert result == {"default": {}}


def test_params_are_not_validated_a_typo_surfaces_at_build_time(tmp_path):
    # The contract is explicit: params.toml is not schema-checked here. A
    # missing key must raise whatever build() itself raises (a plain
    # KeyError), not some earlier validation error invented by load_study.
    _write(tmp_path, "model.py",
           "def build(params):\n    return params['nonexistent_key']\n")
    _write(tmp_path, "params.toml", "mesh = 'fine'\n")
    with pytest.raises(KeyError):
        load_study(tmp_path)


# -- variants() ------------------------------------------------------------


def test_variants_used_when_present(tmp_path):
    _write(tmp_path, "model.py", (
        "def build():\n    return 'never-called'\n\n"
        "def variants():\n    return {'a': 1, 'b': 2}\n"))
    result = load_study(tmp_path)
    assert result == {"a": 1, "b": 2}


def test_variants_only_no_build_at_all(tmp_path):
    _write(tmp_path, "model.py",
           "def variants():\n    return {'only': 'one'}\n")
    result = load_study(tmp_path)
    assert result == {"only": "one"}


def test_variants_as_plain_dict_raises_a_clear_error(tmp_path):
    # A study author might write `variants = {...}` instead of a function.
    # load_study must not call a dict and let a bare "'dict' object is not
    # callable" TypeError leak out -- it should say what's wrong.
    _write(tmp_path, "model.py", (
        "def build():\n    return 'x'\n\n"
        "variants = {'a': 1}\n"))
    with pytest.raises(TypeError, match="variants"):
        load_study(tmp_path)


# -- errors ------------------------------------------------------------


def test_missing_model_py_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="model.py"):
        load_study(tmp_path)


def test_model_py_with_neither_build_nor_variants_raises(tmp_path):
    _write(tmp_path, "model.py", "x = 1\n")
    with pytest.raises(AttributeError, match="build"):
        load_study(tmp_path)


# -- build() signature edge cases ------------------------------------------


def test_build_keyword_only_params_raises_clear_error(tmp_path):
    # Not one of the two supported shapes (build() / build(params)); must
    # not blow up with a confusing low-level "unexpected keyword" TypeError
    # from actually calling it wrong.
    _write(tmp_path, "model.py",
           "def build(*, params=None):\n    return params\n")
    with pytest.raises(TypeError, match="build"):
        load_study(tmp_path)


def test_build_kwargs_only_raises_clear_error(tmp_path):
    _write(tmp_path, "model.py",
           "def build(**kwargs):\n    return kwargs\n")
    with pytest.raises(TypeError, match="build"):
        load_study(tmp_path)


def test_build_two_positional_args_raises_clear_error(tmp_path):
    _write(tmp_path, "model.py",
           "def build(a, b):\n    return a\n")
    with pytest.raises(TypeError, match="build"):
        load_study(tmp_path)


# -- module isolation ------------------------------------------------------


def test_two_studies_with_the_same_directory_name_do_not_collide(tmp_path):
    # Both study dirs are named "study" -- if load_study derived its module
    # name from the basename alone and that name collided in sys.modules,
    # the second load could silently reuse the first module's namespace.
    a = tmp_path / "a" / "study"
    b = tmp_path / "b" / "study"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    _write(a, "model.py", "def build():\n    return 'from-a'\n")
    _write(b, "model.py", "def build():\n    return 'from-b'\n")

    result_a = load_study(a)
    result_b = load_study(b)

    assert result_a == {"default": "from-a"}
    assert result_b == {"default": "from-b"}
