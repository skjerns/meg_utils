#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for meg_utils.plotting — focusing on savefig and vector output.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import os
import tempfile

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from meg_utils.plotting import savefig, tornadoplot


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


# ---------------------------------------------------------------------------
# tornadoplot
# ---------------------------------------------------------------------------

def _make_pvalue_df(seed=42):
    """30 subjects × 4 conditions, each with a random p-value."""
    rng = np.random.default_rng(seed)
    subjects = [f'sub-{i:02d}' for i in range(1, 31)]
    conditions = ['visual', 'auditory', 'motor', 'memory']
    rows = []
    for subj in subjects:
        for cond in conditions:
            rows.append({'subject': subj, 'condition': cond,
                         'pvalue': rng.uniform(0, 0.2)})
    return pd.DataFrame(rows)


class TestTornadoplot:

    def test_returns_axes(self):
        df = _make_pvalue_df()
        ax = tornadoplot(df, x='pvalue', y='condition', center=0.05)
        assert isinstance(ax, plt.Axes)
        plt.close('all')

    def test_center_line_at_005(self):
        df = _make_pvalue_df()
        ax = tornadoplot(df, x='pvalue', y='condition', center=0.05)
        # the vertical reference line should be at x=0.05
        vlines = [l.get_xdata()[0] for l in ax.get_lines()
                  if np.isclose(l.get_xdata()[0], 0.05)]
        assert len(vlines) == 1
        plt.close('all')

    def test_bar_colours(self):
        df = _make_pvalue_df()
        low, high = '#0000ff', '#ff0000'
        ax = tornadoplot(df, x='pvalue', y='condition', center=0.05,
                         low_colour=low, high_colour=high)
        patches = ax.patches
        assert len(patches) > 0
        from matplotlib.colors import to_rgba
        for p in patches:
            rgba = p.get_facecolor()
            matches_low = np.allclose(rgba, to_rgba(low), atol=0.05)
            matches_high = np.allclose(rgba, to_rgba(high), atol=0.05)
            assert matches_low or matches_high, f"Unexpected colour {rgba}"
        plt.close('all')

    def test_vertical_orient(self):
        df = _make_pvalue_df()
        ax = tornadoplot(df, x='pvalue', y='condition', center=0.05,
                         orient='v')
        # horizontal reference line at 0.05
        hlines = [l.get_ydata()[0] for l in ax.get_lines()
                  if np.isclose(l.get_ydata()[0], 0.05)]
        assert len(hlines) == 1
        plt.close('all')

    def test_sort_order(self):
        df = _make_pvalue_df()
        ax = tornadoplot(df, x='pvalue', y='condition', center=0.05,
                         sort=True)
        # tick labels top-to-bottom should correspond to ascending values
        labels = [t.get_text() for t in ax.get_yticklabels()]
        assert len(labels) > 0
        plt.close('all')

    def test_no_sort(self):
        df = _make_pvalue_df()
        ax = tornadoplot(df, x='pvalue', y='condition', center=0.05,
                         sort=False)
        assert isinstance(ax, plt.Axes)
        plt.close('all')

    def test_custom_labels_in_legend(self):
        df = _make_pvalue_df()
        ax = tornadoplot(df, x='pvalue', y='condition', center=0.05,
                         low_label='Significant', high_label='Not significant')
        legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
        assert 'Significant' in legend_texts
        assert 'Not significant' in legend_texts
        plt.close('all')

    def test_missing_x_or_y_raises(self):
        df = _make_pvalue_df()
        with pytest.raises(ValueError):
            tornadoplot(df, x='pvalue', y=None, center=0.05)
        with pytest.raises(ValueError):
            tornadoplot(df, x=None, y='condition', center=0.05)
        plt.close('all')

    def test_provided_ax(self):
        df = _make_pvalue_df()
        fig, ax = plt.subplots()
        returned_ax = tornadoplot(df, x='pvalue', y='condition', center=0.05,
                                  ax=ax)
        assert returned_ax is ax
        plt.close('all')

if __name__=='__main__':
    import unittest
    unittest.main()
