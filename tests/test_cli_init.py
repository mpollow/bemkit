"""`bem init` — bootstrapping a model directory from an existing mesh."""

import meshio
import numpy as np
import pytest

from bemkit.cli import main
from bemkit.config import ModelConfig


def _write_msh(path, vertices, triangles, vibrating=None, names=("rigid", "vibrating")):
    """Write a .msh, optionally tagging some triangles as the driven group.

    ``names`` lets a test mimic a foreign mesher's naming (e.g. Akabak's
    ("HornWalls", "SourceDisc")) while keeping the same two-group scheme.
    """
    if vibrating is None:
        mesh = meshio.Mesh(points=vertices, cells=[("triangle", triangles)])
    else:
        mask = np.zeros(len(triangles), dtype=bool)
        mask[list(vibrating)] = True
        rigid_t, vib_t = triangles[~mask], triangles[mask]
        rigid_name, vib_name = names
        mesh = meshio.Mesh(
            points=vertices,
            cells=[("triangle", rigid_t), ("triangle", vib_t)],
            cell_data={
                "gmsh:physical": [np.full(len(rigid_t), 1), np.full(len(vib_t), 2)],
                "gmsh:geometrical": [np.full(len(rigid_t), 1), np.full(len(vib_t), 2)],
            },
            field_data={rigid_name: np.array([1, 2]), vib_name: np.array([2, 2])},
        )
    meshio.write(str(path), mesh, file_format="gmsh22", binary=False)
    return path


@pytest.fixture()
def bare_msh(tmp_path, tiny_mesh_data):
    """A mesh with no physical groups — the bare-file case init exists for."""
    vertices, triangles = tiny_mesh_data
    d = tmp_path / "src"
    d.mkdir(parents=True, exist_ok=True)
    return _write_msh(d / "bare.msh", vertices, triangles)


@pytest.fixture()
def grouped_msh(tmp_path, tiny_mesh_data):
    """A mesh that already carries a 'vibrating' physical group."""
    vertices, triangles = tiny_mesh_data
    d = tmp_path / "src"
    d.mkdir(parents=True, exist_ok=True)
    return _write_msh(d / "grouped.msh", vertices, triangles, vibrating=[0])


def test_init_writes_a_config_the_parser_accepts(tmp_path, bare_msh):
    model = tmp_path / "model"
    main([str(model), "init", str(bare_msh)])

    config = ModelConfig.from_dir(model)
    assert config.sr == 32000
    assert config.nbins == 65
    assert config.modelname == "model"


def test_init_copies_the_mesh_and_names_the_variant_after_its_stem(tmp_path, bare_msh):
    model = tmp_path / "model"
    main([str(model), "init", str(bare_msh)])

    assert (model / "mesh" / "bare.msh").exists()
    assert ModelConfig.from_dir(model).variants() == ["bare"]


def test_init_without_args_uses_a_mesh_already_in_mesh_dir(tmp_path, tiny_mesh_data):
    model = tmp_path / "model"
    (model / "mesh").mkdir(parents=True)
    vertices, triangles = tiny_mesh_data
    _write_msh(model / "mesh" / "existing.msh", vertices, triangles)

    main([str(model), "init"])

    assert ModelConfig.from_dir(model).variants() == ["existing"]


def test_init_picks_up_a_loose_mesh_in_the_model_dir(tmp_path, tiny_mesh_data):
    model = tmp_path / "model"
    model.mkdir()
    vertices, triangles = tiny_mesh_data
    _write_msh(model / "loose.msh", vertices, triangles)

    main([str(model), "init"])

    assert (model / "mesh" / "loose.msh").exists()
    assert ModelConfig.from_dir(model).variants() == ["loose"]


def test_init_refuses_to_clobber_an_existing_config(tmp_path, bare_msh):
    model = tmp_path / "model"
    main([str(model), "init", str(bare_msh)])
    (model / "config.toml").write_text("# hand-edited\n")

    with pytest.raises(SystemExit):
        main([str(model), "init", str(bare_msh)])
    assert (model / "config.toml").read_text() == "# hand-edited\n"

    main([str(model), "init", str(bare_msh), "--force"])
    assert "[bem]" in (model / "config.toml").read_text()


def test_init_without_any_mesh_exits(tmp_path):
    with pytest.raises(SystemExit):
        main([str(tmp_path / "empty"), "init"])


def test_init_rejects_a_non_mesh_file(tmp_path):
    junk = tmp_path / "notes.txt"
    junk.write_text("hello")
    with pytest.raises(SystemExit):
        main([str(tmp_path / "model"), "init", str(junk)])


def test_groupless_mesh_gets_a_boundary_that_drives_real_elements(tmp_path, bare_msh):
    """The whole point: no 'vibrating' group must not mean a silently rigid model."""
    model = tmp_path / "model"
    main([str(model), "init", str(bare_msh)])

    from bemkit.sim.boundary import flatten, resolve_boundary
    config = ModelConfig.from_dir(model)
    assert config.boundary_cfg, "no BOUNDARY table for a group-less mesh"

    resolved = resolve_boundary(config.boundary_cfg, "bare",
                                model / "mesh" / "bare.msh",
                                default_velocity=config.default_velocity)
    assert flatten(resolved), "synthesised anchor selected no elements"


def test_mesh_with_vibrating_group_gets_no_boundary(tmp_path, grouped_msh):
    model = tmp_path / "model"
    main([str(model), "init", str(grouped_msh)])
    assert ModelConfig.from_dir(model).get("BOUNDARY") is None


@pytest.fixture()
def foreign_msh(tmp_path, tiny_mesh_data):
    """Right scheme, foreign names — the Akabak horn case (SourceDisc/HornWalls)."""
    vertices, triangles = tiny_mesh_data
    d = tmp_path / "src"
    d.mkdir(parents=True, exist_ok=True)
    return _write_msh(d / "horn.msh", vertices, triangles, vibrating=[0],
                      names=("HornWalls", "SourceDisc"))


def test_foreign_group_names_are_named_in_config_not_guessed_geometrically(
        tmp_path, foreign_msh):
    """The horn case: a driven group exists but isn't called 'vibrating'."""
    model = tmp_path / "model"
    main([str(model), "init", str(foreign_msh)])

    config = ModelConfig.from_dir(model)
    assert config.vibrating_group() == "SourceDisc"
    # A geometric placeholder here would silently drive the wrong patch.
    assert config.get("BOUNDARY") is None


def test_foreign_group_names_export_to_the_right_elements(tmp_path, foreign_msh):
    from bemkit.pipeline import STAGES

    model = tmp_path / "model"
    main([str(model), "init", str(foreign_msh)])
    STAGES["export"].run(ModelConfig.from_dir(model))

    body = (model / "numcalc" / "horn" / "NC.inp").read_text()
    boundary = body.split("BOUNDARY", 1)[1].split("RETU", 1)[0]
    # meshio writes the rigid block first, so the single SourceDisc triangle
    # ends up at index 3. Exactly that element must be driven — nothing else.
    assert boundary.strip() == "ELEM 3 TO 3 VELO 1.0000 -1 0.0 -1"


def test_explicit_group_name_that_is_absent_raises(tmp_path, foreign_msh):
    """A typo'd group name must fail, not fall back to a rigid surface."""
    from bemkit.meshing.io import (
        read_mesh_physical_groups, resolve_vibrating_group,
    )
    groups = read_mesh_physical_groups(foreign_msh)
    with pytest.raises(ValueError, match="not a physical group"):
        resolve_vibrating_group(groups, "Typo")


def test_vibrating_group_lookup_is_case_insensitive(tmp_path, tiny_mesh_data):
    from bemkit.meshing.io import (
        read_mesh_physical_groups, resolve_vibrating_group,
    )
    vertices, triangles = tiny_mesh_data
    path = _write_msh(tmp_path / "cap.msh", vertices, triangles, vibrating=[0],
                      names=("Rigid", "Vibrating"))
    name, idx = resolve_vibrating_group(read_mesh_physical_groups(path))
    assert name == "Vibrating" and len(idx) == 1


def test_groups_section_rejects_unknown_keys(tmp_path, foreign_msh):
    model = tmp_path / "model"
    main([str(model), "init", str(foreign_msh)])
    cfg = model / "config.toml"
    cfg.write_text(cfg.read_text() + '\nrigid = "HornWalls"\n')
    with pytest.raises(ValueError, match=r"unknown \[groups\] key"):
        ModelConfig.from_dir(model)


def test_init_then_export_produces_a_runnable_bundle(tmp_path, bare_msh):
    """Catches fieldpoint radii that land inside the mesh and get dropped."""
    from bemkit.pipeline import STAGES

    model = tmp_path / "model"
    main([str(model), "init", str(bare_msh)])
    STAGES["export"].run(ModelConfig.from_dir(model))

    nc_inp = model / "numcalc" / "bare" / "NC.inp"
    assert nc_inp.exists()
    body = nc_inp.read_text()
    boundary = body.split("BOUNDARY", 1)[1].split("RETU", 1)[0]
    assert "VELO" in boundary, "exported bundle has no driven elements"
