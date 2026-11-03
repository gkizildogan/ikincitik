"""Section-bounded, structure-preserving extraction of EASA CS-E Amendment 9."""

from .segment import AMDT_RE, Region, Section, build_sections, coverage_report, load_sections, save_sections

__all__ = ["AMDT_RE", "Region", "Section", "build_sections", "coverage_report", "load_sections", "save_sections"]
