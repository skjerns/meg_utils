#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for meg_utils.plotting — focusing on savefig and vector output.
"""
import sys; sys.path.append('../..')

import json
import os
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pytest
from PIL import Image

from meg_utils.plotting import savefig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_fig():
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])
    return fig


# ---------------------------------------------------------------------------
# savefig — basic file creation
# ---------------------------------------------------------------------------

class TestSavefig:

    def test_saves_png(self, tmp_path):
        fig = _simple_fig()
        out = str(tmp_path / 'plot.png')
        savefig(fig, out, metadata=False)
        assert os.path.exists(out)
        plt.close('all')

    def test_default_extension_added(self, tmp_path):
        fig = _simple_fig()
        out = str(tmp_path / 'plot')
        savefig(fig, out, metadata=False)
        assert os.path.exists(out + '.png')
        plt.close('all')

    def test_saves_jpg(self, tmp_path):
        fig = _simple_fig()
        out = str(tmp_path / 'plot.jpg')
        savefig(fig, out, metadata=False)
        assert os.path.exists(out)
        plt.close('all')

    def test_saves_svg(self, tmp_path):
        fig = _simple_fig()
        out = str(tmp_path / 'plot.svg')
        savefig(fig, out, metadata=False)
        assert os.path.exists(out)
        plt.close('all')

    def test_creates_output_directory(self, tmp_path):
        fig = _simple_fig()
        out = str(tmp_path / 'subdir' / 'deep' / 'plot.png')
        savefig(fig, out, metadata=False)
        assert os.path.exists(out)
        plt.close('all')


# ---------------------------------------------------------------------------
# savefig — vector output (save_vector=True)
# ---------------------------------------------------------------------------

class TestSavefigVector:

    def test_vector_creates_vectors_subdir(self, tmp_path):
        fig = _simple_fig()
        out = str(tmp_path / 'plot.png')
        savefig(fig, out, save_vector=True, metadata=False)
        assert os.path.isdir(str(tmp_path / 'vectors'))
        plt.close('all')

    def test_vector_creates_svg(self, tmp_path):
        fig = _simple_fig()
        out = str(tmp_path / 'plot.png')
        savefig(fig, out, save_vector=True, metadata=False)
        assert os.path.exists(str(tmp_path / 'vectors' / 'plot.svg'))
        plt.close('all')

    def test_vector_creates_eps(self, tmp_path):
        fig = _simple_fig()
        out = str(tmp_path / 'plot.png')
        savefig(fig, out, save_vector=True, metadata=False)
        assert os.path.exists(str(tmp_path / 'vectors' / 'plot.eps'))
        plt.close('all')

    def test_vector_disabled(self, tmp_path):
        fig = _simple_fig()
        out = str(tmp_path / 'plot.png')
        savefig(fig, out, save_vector=False, metadata=False)
        assert not os.path.isdir(str(tmp_path / 'vectors'))
        plt.close('all')

    def test_vector_basename_matches_source(self, tmp_path):
        fig = _simple_fig()
        out = str(tmp_path / 'my_figure.png')
        savefig(fig, out, save_vector=True, metadata=False)
        assert os.path.exists(str(tmp_path / 'vectors' / 'my_figure.svg'))
        assert os.path.exists(str(tmp_path / 'vectors' / 'my_figure.eps'))
        plt.close('all')

    def test_vector_dpi_kwarg_excluded(self, tmp_path):
        """dpi should not be passed to vector formats (no error raised)."""
        fig = _simple_fig()
        out = str(tmp_path / 'plot.png')
        savefig(fig, out, save_vector=True, metadata=False, dpi=300)
        assert os.path.exists(str(tmp_path / 'vectors' / 'plot.svg'))
        plt.close('all')

    def test_vector_metadata_json_written(self, tmp_path):
        fig = _simple_fig()
        out = str(tmp_path / 'plot.png')
        savefig(fig, out, save_vector=True, metadata={'key': 'value'})
        json_file = str(tmp_path / 'vectors' / 'plot.json')
        assert os.path.exists(json_file)
        with open(json_file) as f:
            data = json.load(f)
        assert data['key'] == 'value'
        plt.close('all')

    def test_vector_no_json_when_metadata_false(self, tmp_path):
        fig = _simple_fig()
        out = str(tmp_path / 'plot.png')
        savefig(fig, out, save_vector=True, metadata=False)
        json_file = str(tmp_path / 'vectors' / 'plot.json')
        assert not os.path.exists(json_file)
        plt.close('all')

    def test_vector_svg_for_svg_source(self, tmp_path):
        """When saving an SVG directly, vectors/ still gets SVG and EPS copies."""
        fig = _simple_fig()
        out = str(tmp_path / 'plot.svg')
        savefig(fig, out, save_vector=True, metadata=False)
        assert os.path.exists(str(tmp_path / 'vectors' / 'plot.svg'))
        assert os.path.exists(str(tmp_path / 'vectors' / 'plot.eps'))
        plt.close('all')


# ---------------------------------------------------------------------------
# savefig — metadata embedding
# ---------------------------------------------------------------------------

class TestSavefigMetadata:

    def test_png_metadata_dict(self, tmp_path):
        fig = _simple_fig()
        out = str(tmp_path / 'plot.png')
        savefig(fig, out, metadata={'author': 'test', 'experiment': 'MEG'})
        img = Image.open(out)
        assert img.text.get('author') == 'test'
        assert img.text.get('experiment') == 'MEG'
        img.close()
        plt.close('all')

    def test_png_metadata_string(self, tmp_path):
        fig = _simple_fig()
        out = str(tmp_path / 'plot.png')
        savefig(fig, out, metadata='my note')
        img = Image.open(out)
        assert img.text.get('metadata') == 'my note'
        img.close()
        plt.close('all')

    def test_png_metadata_false_no_chunks(self, tmp_path):
        fig = _simple_fig()
        out = str(tmp_path / 'plot.png')
        savefig(fig, out, metadata=False)
        img = Image.open(out)
        # No custom text chunks should be present
        assert 'author' not in img.text
        img.close()
        plt.close('all')

    def test_png_metadata_none_auto(self, tmp_path):
        """metadata=None should auto-generate at least script_path."""
        fig = _simple_fig()
        out = str(tmp_path / 'plot.png')
        savefig(fig, out, save_vector=False, metadata=None)
        img = Image.open(out)
        assert 'script_path' in img.text
        img.close()
        plt.close('all')
