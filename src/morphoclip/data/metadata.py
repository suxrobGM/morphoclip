"""CPJUMP1 metadata index: unified (plate, well) -> PerturbationInfo lookup.

Parses plate maps, barcode mappings, and external metadata TSVs,
then caches the full lookup for fast access during training.
"""

import csv
import logging
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

from morphoclip.data.perturbation import PerturbationInfo, PerturbationType

logger = logging.getLogger(__name__)


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _classify_pert_type(
    row: dict[str, str],
    compound_ids: set[str],
    crispr_ids: set[str],
    orf_ids: set[str],
) -> PerturbationType:
    """Determine perturbation type from a platemap row."""
    broad_sample = row.get("broad_sample", "")
    pert_type_str = row.get("pert_type", "").lower()
    control_type = row.get("control_type", "").lower()

    if "negcon" in control_type:
        return PerturbationType.NEGCON
    if "poscon" in control_type:
        return PerturbationType.POSCON
    if pert_type_str == "control" or not broad_sample:
        return PerturbationType.NEGCON

    if broad_sample in compound_ids:
        return PerturbationType.COMPOUND
    if broad_sample in crispr_ids:
        return PerturbationType.CRISPR
    if broad_sample in orf_ids:
        return PerturbationType.ORF

    if broad_sample.startswith("BRD-"):
        return PerturbationType.COMPOUND
    return PerturbationType.UNKNOWN


class MetadataIndex:
    """Metadata index for CPJUMP1.

    Merges plate maps, barcode mappings, and external metadata into a
    single lookup: ``(plate_barcode, well) -> PerturbationInfo``.

    Usage::

        index = MetadataIndex.from_config(Path("configs/dataset.yml"))
        info = index.lookup("BR00116991", "A01")
    """

    def __init__(
        self,
        barcode_to_platemap: dict[str, str],
        platemap_wells: dict[str, dict[str, dict[str, str]]],
        compound_meta: dict[str, dict[str, str]],
        crispr_meta: dict[str, dict[str, str]],
        orf_meta: dict[str, dict[str, str]],
    ) -> None:
        self._barcode_to_platemap = barcode_to_platemap
        self._platemap_wells = platemap_wells
        self._compound_meta = compound_meta
        self._crispr_meta = crispr_meta
        self._orf_meta = orf_meta

        self._compound_ids = set(compound_meta.keys())
        self._crispr_ids = set(crispr_meta.keys())
        self._orf_ids = set(orf_meta.keys())

        self._cache: dict[tuple[str, str], PerturbationInfo] = {}
        self._build_cache()

    def _build_cache(self) -> None:
        for barcode, platemap_name in self._barcode_to_platemap.items():
            wells = self._platemap_wells.get(platemap_name, {})
            for well, row in wells.items():
                self._cache[(barcode, well)] = self._build_info(row)

    def _build_info(self, row: dict[str, str]) -> PerturbationInfo:
        broad_sample = row.get("broad_sample", "")
        pert_type = _classify_pert_type(row, self._compound_ids, self._crispr_ids, self._orf_ids)
        base = PerturbationInfo(
            pert_type=pert_type,
            broad_sample=broad_sample,
            control_type=row.get("control_type", ""),
        )

        if pert_type == PerturbationType.COMPOUND and broad_sample in self._compound_meta:
            meta = self._compound_meta[broad_sample]
            return replace(
                base,
                pert_iname=meta.get("pert_iname", ""),
                target_list=meta.get("target", meta.get("target_list", meta.get("gene", ""))),
                smiles=meta.get("smiles", ""),
                pubchem_cid=meta.get("pubchem_cid", ""),
                moa=meta.get("moa", ""),
            )

        if pert_type == PerturbationType.CRISPR and broad_sample in self._crispr_meta:
            meta = self._crispr_meta[broad_sample]
            return replace(
                base,
                gene=meta.get("gene", ""),
                pert_iname=meta.get("pert_iname", meta.get("gene", "")),
                target_sequence=meta.get("target_sequence", ""),
                negcon_control_type=meta.get("negcon_control_type", ""),
            )

        if pert_type == PerturbationType.ORF and broad_sample in self._orf_meta:
            meta = self._orf_meta[broad_sample]
            return replace(
                base,
                gene=meta.get("gene", ""),
                pert_iname=meta.get("pert_iname", meta.get("gene", "")),
                negcon_control_type=meta.get("negcon_control_type", ""),
            )

        return base

    @classmethod
    def from_config(cls, config_path: Path) -> MetadataIndex:
        """Build index from ``configs/dataset.yml``."""
        from morphoclip.data.config import load_dataset_config

        config = load_dataset_config(config_path)
        return cls.from_directory(config.local.metadata, config.batch)

    @classmethod
    def from_directory(cls, metadata_dir: Path, batch: str | None = None) -> MetadataIndex:
        """Build index from a metadata directory tree."""
        platemap_dir = metadata_dir / "platemaps"
        ext_dir = metadata_dir / "external_metadata"

        if batch is None:
            batch_dirs = [d for d in platemap_dir.iterdir() if d.is_dir()]
            if not batch_dirs:
                raise FileNotFoundError(f"No batch directories in {platemap_dir}")
            batch = batch_dirs[0].name
            logger.info("Auto-detected batch: %s", batch)

        batch_dir = platemap_dir / batch

        barcode_to_platemap = cls._load_barcode_map(batch_dir)
        platemap_wells = cls._load_platemaps(batch_dir)

        compound_targets = ext_dir / "JUMP-Target-1_compound_metadata_targets.tsv"
        compound_file = (
            "JUMP-Target-1_compound_metadata_targets.tsv"
            if compound_targets.exists()
            else "JUMP-Target-1_compound_metadata.tsv"
        )
        compound_meta = cls._load_external(ext_dir, compound_file)
        crispr_meta = cls._load_external(ext_dir, "JUMP-Target-1_crispr_metadata.tsv")
        orf_meta = cls._load_external(ext_dir, "JUMP-Target-1_orf_metadata.tsv")

        return cls(
            barcode_to_platemap=barcode_to_platemap,
            platemap_wells=platemap_wells,
            compound_meta=compound_meta,
            crispr_meta=crispr_meta,
            orf_meta=orf_meta,
        )

    @staticmethod
    def _load_barcode_map(batch_dir: Path) -> dict[str, str]:
        barcode_csv = batch_dir / "barcode_platemap.csv"

        if not barcode_csv.exists():
            logger.warning("barcode_platemap.csv not found at %s", barcode_csv)
            return {}

        result: dict[str, str] = {}

        for row in _read_csv(barcode_csv):
            barcode = row.get("Assay_Plate_Barcode", "")
            platemap_name = row.get("Plate_Map_Name", "")
            if barcode and platemap_name:
                result[barcode] = platemap_name

        logger.info("Loaded %d barcode mappings", len(result))
        return result

    @staticmethod
    def _load_platemaps(batch_dir: Path) -> dict[str, dict[str, dict[str, str]]]:
        txt_files = sorted(batch_dir.glob("*.txt"))
        platemap_subdir = batch_dir / "platemap"
        if platemap_subdir.is_dir():
            txt_files = sorted(set(txt_files) | set(platemap_subdir.glob("*.txt")))

        result: dict[str, dict[str, dict[str, str]]] = {}
        for txt_file in txt_files:
            wells: dict[str, dict[str, str]] = {}

            for row in _read_tsv(txt_file):
                well = row.get("well_position", "")
                if well:
                    wells[well] = dict(row)

            result[txt_file.stem] = wells
            logger.debug("Loaded platemap %s with %d wells", txt_file.stem, len(wells))

        logger.info("Loaded %d platemap files", len(result))
        return result

    @staticmethod
    def _load_external(ext_dir: Path, filename: str) -> dict[str, dict[str, str]]:
        path = ext_dir / filename
        if not path.exists():
            logger.warning("External metadata not found: %s", path)
            return {}
        result: dict[str, dict[str, str]] = {}
        for row in _read_tsv(path):
            key = row.get("broad_sample", "")
            if key:
                result[key] = dict(row)
        logger.info("Loaded %d entries from %s", len(result), filename)
        return result

    def lookup(self, plate_barcode: str, well: str) -> PerturbationInfo:
        """Look up perturbation info for a (plate, well) pair."""
        well = well.upper()
        info = self._cache.get((plate_barcode, well))
        if info is not None:
            return info
        logger.debug("No metadata for plate=%s well=%s", plate_barcode, well)
        return PerturbationInfo()

    def wells_for_plate(self, plate_barcode: str) -> list[str]:
        """List all well positions for a given plate."""
        platemap_name = self._barcode_to_platemap.get(plate_barcode, "")
        return sorted(self._platemap_wells.get(platemap_name, {}).keys())

    def plates(self) -> list[str]:
        """List all plate barcodes in the index."""
        return sorted(self._barcode_to_platemap.keys())

    def perturbation_infos(self) -> Iterator[PerturbationInfo]:
        """Yield the ``PerturbationInfo`` of every indexed (plate, well), in index order."""
        yield from self._cache.values()

    def unique_perturbations(self) -> list[PerturbationInfo]:
        """One ``PerturbationInfo`` per distinct perturbation, first occurrence wins.

        Perturbations are keyed by ``broad_sample``, or by type and name when a
        well has none (controls).
        """
        seen: set[str] = set()
        result: list[PerturbationInfo] = []
        for info in self.perturbation_infos():
            key = info.broad_sample or f"{info.pert_type.value}:{info.pert_iname or info.gene}"
            if key not in seen:
                seen.add(key)
                result.append(info)
        return result

    def __len__(self) -> int:
        return len(self._cache)

    def __repr__(self) -> str:
        return (
            f"MetadataIndex(plates={len(self._barcode_to_platemap)}, "
            f"wells={len(self._cache)}, "
            f"compounds={len(self._compound_meta)}, "
            f"crispr={len(self._crispr_meta)}, "
            f"orf={len(self._orf_meta)})"
        )
