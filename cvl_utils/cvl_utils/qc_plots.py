"""
qc_plots.py
===========
Lightweight interactive QC figures (plotly) for pipeline outputs.
"""

import numpy as np
import nibabel as nib
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_AXIS = {'sag': 0, 'cor': 1, 'ax': 2}


def _canonical_data(path: str) -> np.ndarray:
    img = nib.as_closest_canonical(nib.load(path))
    return np.asanyarray(img.dataobj, dtype=np.float32)


def sdc_qc_plot(
    orig_path: str,
    sdc_path: str,
    plane: str = 'sag',
    title: str = None,
    pct_clip: float = 99.5,
):
    """
    Interactive before/after QC for susceptibility distortion correction.

    Shows the distorted input, the SDC-corrected output, their voxelwise
    difference, and a "flicker" panel that blinks between the two (press
    the Blink button) so the contrast/shape change is easy to spot, for one
    orthogonal plane, with a slider to scroll through slices. *orig_path*
    and *sdc_path* must be on the same grid (same shape + affine), which is
    the case for the AFNI SDC pipeline outputs.

    Default plane is sagittal ('sag') since phase-encode distortion is
    typically along the anterior-posterior axis, which sagittal slices show
    in-plane (other options: 'cor', 'ax'). The flicker panel blinks at
    whichever slice is displayed when the slider was last moved (it does
    not itself move the slider).
    """
    orig = _canonical_data(orig_path)
    sdc = _canonical_data(sdc_path)
    if orig.shape != sdc.shape:
        raise ValueError(
            'sdc_qc_plot: shape mismatch, orig {} vs sdc {}'.format(orig.shape, sdc.shape))

    axis = _AXIS[plane]
    n_slices = orig.shape[axis]
    diff = sdc - orig

    vmax = np.percentile(orig, pct_clip)
    dmax = np.percentile(np.abs(diff), pct_clip) or 1.0

    def _slice(vol, i):
        return np.rot90(np.take(vol, i, axis=axis))

    common = dict(zmin=0, zmax=vmax, colorscale='gray', showscale=False)
    frames = [
        go.Frame(
            name=str(i),
            traces=[0, 1, 2],
            data=[
                go.Heatmap(z=_slice(orig, i), **common),
                go.Heatmap(z=_slice(sdc, i), **common),
                go.Heatmap(z=_slice(diff, i), zmin=-dmax, zmax=dmax,
                           colorscale='RdBu', showscale=False),
            ],
        )
        for i in range(n_slices)
    ]

    mid = n_slices // 2
    # Flicker frames toggle only the 4th panel (trace index 3) between the
    # uncorrected and corrected slice; kept separate from the slice frames
    # above so the slider and the blink button don't fight over trace 3.
    flicker_frames = [
        go.Frame(name='flicker-orig', traces=[3], data=[go.Heatmap(z=_slice(orig, mid), **common)]),
        go.Frame(name='flicker-sdc', traces=[3], data=[go.Heatmap(z=_slice(sdc, mid), **common)]),
    ]

    fig = make_subplots(
        rows=1, cols=4,
        subplot_titles=('distorted (pre-SDC)', 'corrected (SDC)', 'difference (post − pre)',
                         'flicker (blink compare)'),
    )
    for col, trace in enumerate(frames[mid].data, start=1):
        fig.add_trace(trace, row=1, col=col)
    fig.add_trace(go.Heatmap(z=_slice(orig, mid), **common), row=1, col=4)
    fig.frames = frames + flicker_frames

    n_blinks = 20
    blink_sequence = ['flicker-orig', 'flicker-sdc'] * n_blinks
    fig.update_layout(
        title=title or 'SDC QC ({} plane)'.format(plane),
        height=380,
        width=1350,
        margin=dict(t=60, b=60, l=20, r=20),
        sliders=[dict(
            active=mid,
            currentvalue=dict(prefix='slice: '),
            steps=[
                dict(method='animate', label=str(i),
                     args=[[str(i)], dict(mode='immediate',
                                           frame=dict(duration=0, redraw=True))])
                for i in range(n_slices)
            ],
        )],
        updatemenus=[dict(
            type='buttons',
            direction='left',
            x=1.0, y=1.18, xanchor='right', yanchor='top',
            showactive=False,
            buttons=[
                dict(label='▶ Blink', method='animate',
                     args=[blink_sequence, dict(
                         frame=dict(duration=400, redraw=True),
                         transition=dict(duration=0),
                         fromcurrent=False, mode='immediate')]),
                dict(label='⏸ Stop', method='animate',
                     args=[[None], dict(mode='immediate',
                                         frame=dict(duration=0, redraw=False))]),
            ],
        )],
    )
    for ax_name in ('xaxis', 'xaxis2', 'xaxis3', 'xaxis4'):
        fig.update_layout({ax_name: dict(visible=False)})
    for ax_name in ('yaxis', 'yaxis2', 'yaxis3', 'yaxis4'):
        fig.update_layout({ax_name: dict(visible=False, scaleanchor=ax_name.replace('yaxis', 'x'))})

    return fig


def motion_qc_plot(
    par_file: str,
    tr: float = None,
    fd_thresh: float = 0.5,
    head_radius_mm: float = 50.0,
    title: str = None,
):
    """
    Interactive QC for MCFLIRT motion parameters.

    Reads a 6-column FSL `.par` file (rot_x, rot_y, rot_z [rad], trans_x,
    trans_y, trans_z [mm]) and plots translations, rotations, and framewise
    displacement (FD, Power et al. convention: rotations converted to mm
    using *head_radius_mm*) as three linked, zoomable/hoverable timeseries.
    A dashed line marks *fd_thresh*; volumes above it are flagged in the
    title.
    """
    data = np.loadtxt(par_file)
    rot_rad, trans = data[:, :3], data[:, 3:]
    rot_deg = np.degrees(rot_rad)

    fd = np.concatenate([[0.0], np.abs(np.diff(
        np.hstack([trans, rot_rad * head_radius_mm]), axis=0)).sum(axis=1)])

    n_vols = data.shape[0]
    x = np.arange(n_vols) * tr if tr else np.arange(n_vols)
    xlabel = 'time (s)' if tr else 'volume'

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        subplot_titles=('translation (mm)', 'rotation (deg)', 'framewise displacement (mm)'),
    )
    for col, label in zip(range(3), ('x', 'y', 'z')):
        fig.add_trace(go.Scatter(x=x, y=trans[:, col], name='trans_{}'.format(label),
                                 mode='lines'), row=1, col=1)
    for col, label in zip(range(3), ('x', 'y', 'z')):
        fig.add_trace(go.Scatter(x=x, y=rot_deg[:, col], name='rot_{}'.format(label),
                                 mode='lines'), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=fd, name='FD', mode='lines',
                             line=dict(color='black')), row=3, col=1)

    outliers = fd > fd_thresh
    if outliers.any():
        fig.add_trace(go.Scatter(x=x[outliers], y=fd[outliers], name='FD > {}mm'.format(fd_thresh),
                                 mode='markers', marker=dict(color='red', size=7)), row=3, col=1)
    fig.add_hline(y=fd_thresh, line=dict(color='red', dash='dash'), row=3, col=1)

    fig.update_xaxes(title_text=xlabel, row=3, col=1)
    fig.update_layout(
        title=title or 'Motion QC — {} volumes, {}/{} above {}mm FD'.format(
            n_vols, int(outliers.sum()), n_vols, fd_thresh),
        height=650,
        width=950,
        margin=dict(t=80, b=60, l=60, r=20),
    )
    return fig


_ORTHO_DIV_ID = 'ortho_view_plot'

_ORTHO_CLICK_JS = """
var gd = document.getElementById('{div_id}');
var NX = {nx}, NY = {ny}, NZ = {nz};
var ci = {i0}, cj = {j0}, ck = {k0};
function clamp(v, n) {{ return Math.max(0, Math.min(n - 1, Math.round(v))); }}
function moveCrosshair(i, j, k) {{
  ci = i; cj = j; ck = k;
  Plotly.animate(gd, ['sag-' + i, 'cor-' + j, 'ax-' + k],
    {{frame: {{duration: 0, redraw: true}}, transition: {{duration: 0}}, mode: 'immediate'}}
  ).then(function() {{
    Plotly.relayout(gd, {{
      'shapes[0].x0': k, 'shapes[0].x1': k,
      'shapes[1].y0': j, 'shapes[1].y1': j,
      'shapes[2].x0': k, 'shapes[2].x1': k,
      'shapes[3].y0': i, 'shapes[3].y1': i,
      'shapes[4].x0': j, 'shapes[4].x1': j,
      'shapes[5].y0': i, 'shapes[5].y1': i
    }});
  }});
}}
gd.on('plotly_click', function(evt) {{
  var pt = evt.points[0], curve = pt.curveNumber;
  var col = pt.x, row = pt.y, i, j, k;
  if (curve === 0) {{ i = ci; k = clamp(col, NZ); j = clamp(row, NY); }}
  else if (curve === 1) {{ j = cj; k = clamp(col, NZ); i = clamp(row, NX); }}
  else if (curve === 2) {{ k = ck; j = clamp(col, NY); i = clamp(row, NX); }}
  else return;
  moveCrosshair(i, j, k);
}});
"""


def ortho_view_plot(
    vol,
    crosshair: tuple = None,
    pct_clip: float = 99.5,
    title: str = None,
):
    """
    Traditional 3-view (sagittal / coronal / axial) volume viewer with a
    clickable crosshair, built in plain Plotly — no server, no extra
    dependency beyond what's already used in this module. Not MRI-specific:
    works on any 3D (or 4D) array with axes ordered (x, y, z[, t]).

    Click any panel to move the crosshair; the other two panels jump to the
    matching slice, and the crosshair lines update on all three. For 4D
    input a time slider is added, but note it scrubs the timeseries at the
    crosshair position the figure was BUILT with (`crosshair`, default:
    volume centre) — spatial navigation runs against the temporal mean, and
    clicking after moving the time slider snaps the other panels back to
    that fixed spatial location rather than tracking the click across time
    too. Fully live crosshair+time together would need either a server
    (Dash) or embedding the whole 4D array in the HTML, which defeats the
    point of a lightweight, report-able static file.

    Parameters
    ----------
    vol : str or np.ndarray
        Path to a nibabel-loadable image, or an (x, y, z[, t]) array.
    crosshair : (i, j, k), optional
        Initial/fixed crosshair voxel indices. Defaults to the volume centre.

    Returns
    -------
    fig, post_script : the Plotly figure and a JS snippet implementing the
        click handler. Export with, e.g.::

            fig, post_script = ortho_view_plot('mean_func.nii.gz')
            fig.write_html('qc_report.html', div_id='ortho_view_plot',
                            post_script=post_script)

        `div_id` must match the module constant `_ORTHO_DIV_ID` baked into
        `post_script` (the default shown above is correct).
    """
    if isinstance(vol, str):
        data = _canonical_data(vol)
    else:
        data = np.asarray(vol, dtype=np.float32)
    if data.ndim not in (3, 4):
        raise ValueError('ortho_view_plot: expected a 3D or 4D array, got shape {}'.format(data.shape))

    is4d = data.ndim == 4
    ref = data.mean(axis=3) if is4d else data
    nx, ny, nz = ref.shape[:3]
    i0, j0, k0 = crosshair if crosshair else (nx // 2, ny // 2, nz // 2)

    vmax = np.percentile(ref, pct_clip)
    common = dict(zmin=0, zmax=vmax, colorscale='gray', showscale=False)

    def sag(i): return ref[i, :, :]
    def cor(j): return ref[:, j, :]
    def ax(k): return ref[:, :, k]

    frames = (
        [go.Frame(name='sag-{}'.format(i), traces=[0], data=[go.Heatmap(z=sag(i), **common)]) for i in range(nx)]
        + [go.Frame(name='cor-{}'.format(j), traces=[1], data=[go.Heatmap(z=cor(j), **common)]) for j in range(ny)]
        + [go.Frame(name='ax-{}'.format(k), traces=[2], data=[go.Heatmap(z=ax(k), **common)]) for k in range(nz)]
    )

    if is4d:
        nt = data.shape[3]
        frames += [
            go.Frame(
                name='t-{}'.format(t), traces=[0, 1, 2],
                data=[
                    go.Heatmap(z=data[i0, :, :, t], **common),
                    go.Heatmap(z=data[:, j0, :, t], **common),
                    go.Heatmap(z=data[:, :, k0, t], **common),
                ],
            )
            for t in range(nt)
        ]

    fig = make_subplots(rows=1, cols=3, subplot_titles=('sagittal', 'coronal', 'axial'))
    fig.add_trace(go.Heatmap(z=sag(i0), **common), row=1, col=1)
    fig.add_trace(go.Heatmap(z=cor(j0), **common), row=1, col=2)
    fig.add_trace(go.Heatmap(z=ax(k0), **common), row=1, col=3)
    fig.frames = frames

    line = dict(color='lime', width=1)
    fig.update_layout(
        title=title or 'ortho view',
        height=380,
        width=1100,
        margin=dict(t=60, b=40, l=20, r=20),
        shapes=[
            dict(type='line', xref='x', yref='y', x0=k0, x1=k0, y0=0, y1=ny - 1, line=line),
            dict(type='line', xref='x', yref='y', x0=0, x1=nz - 1, y0=j0, y1=j0, line=line),
            dict(type='line', xref='x2', yref='y2', x0=k0, x1=k0, y0=0, y1=nx - 1, line=line),
            dict(type='line', xref='x2', yref='y2', x0=0, x1=nz - 1, y0=i0, y1=i0, line=line),
            dict(type='line', xref='x3', yref='y3', x0=j0, x1=j0, y0=0, y1=nx - 1, line=line),
            dict(type='line', xref='x3', yref='y3', x0=0, x1=ny - 1, y0=i0, y1=i0, line=line),
        ],
    )
    for ax_name in ('xaxis', 'xaxis2', 'xaxis3'):
        fig.update_layout({ax_name: dict(visible=False)})
    for ax_name in ('yaxis', 'yaxis2', 'yaxis3'):
        fig.update_layout({ax_name: dict(visible=False, scaleanchor=ax_name.replace('yaxis', 'x'))})

    if is4d:
        fig.update_layout(sliders=[dict(
            active=0,
            currentvalue=dict(prefix='volume: '),
            steps=[
                dict(method='animate', label=str(t),
                     args=[['t-{}'.format(t)], dict(mode='immediate', frame=dict(duration=0, redraw=True))])
                for t in range(nt)
            ],
        )])

    post_script = _ORTHO_CLICK_JS.format(div_id=_ORTHO_DIV_ID, nx=nx, ny=ny, nz=nz, i0=i0, j0=j0, k0=k0)
    return fig, post_script


_SDC_ORTHO_DIV_ID = 'sdc_ortho_qc_plot'

# Fixed shape-index layout (24 crosshair lines: 3 rows x 4 panels x {v,h}),
# independent of volume shape, so it can be hardcoded here rather than
# templated per call.
_SDC_ORTHO_CLICK_JS = """
var gd = document.getElementById('{div_id}');
var NX = {nx}, NY = {ny}, NZ = {nz};
var ci = {i0}, cj = {j0}, ck = {k0};
var blinkState = 0, blinkTimer = null;
function clamp(v, n) {{ return Math.max(0, Math.min(n - 1, Math.round(v))); }}
function render() {{
  var suf = blinkState ? 'sdc' : 'orig';
  Plotly.animate(gd, ['sag-' + ci + '-' + suf, 'cor-' + cj + '-' + suf, 'ax-' + ck + '-' + suf],
    {{frame: {{duration: 0, redraw: true}}, transition: {{duration: 0}}, mode: 'immediate'}});
}}
function updateShapes(i, j, k) {{
  var upd = {{}};
  [0, 2, 4, 6].forEach(function(n) {{ upd['shapes[' + n + '].x0'] = k; upd['shapes[' + n + '].x1'] = k; }});
  [1, 3, 5, 7].forEach(function(n) {{ upd['shapes[' + n + '].y0'] = j; upd['shapes[' + n + '].y1'] = j; }});
  [8, 10, 12, 14].forEach(function(n) {{ upd['shapes[' + n + '].x0'] = k; upd['shapes[' + n + '].x1'] = k; }});
  [9, 11, 13, 15].forEach(function(n) {{ upd['shapes[' + n + '].y0'] = i; upd['shapes[' + n + '].y1'] = i; }});
  [16, 18, 20, 22].forEach(function(n) {{ upd['shapes[' + n + '].x0'] = j; upd['shapes[' + n + '].x1'] = j; }});
  [17, 19, 21, 23].forEach(function(n) {{ upd['shapes[' + n + '].y0'] = i; upd['shapes[' + n + '].y1'] = i; }});
  Plotly.relayout(gd, upd);
}}
gd.on('plotly_click', function(evt) {{
  var pt = evt.points[0], curve = pt.curveNumber;
  var col = pt.x, row = pt.y, i, j, k;
  if (curve >= 0 && curve <= 3) {{ i = ci; k = clamp(col, NZ); j = clamp(row, NY); }}
  else if (curve >= 4 && curve <= 7) {{ j = cj; k = clamp(col, NZ); i = clamp(row, NX); }}
  else if (curve >= 8 && curve <= 11) {{ k = ck; j = clamp(col, NY); i = clamp(row, NX); }}
  else return;
  ci = i; cj = j; ck = k;
  render();
  updateShapes(i, j, k);
}});
gd.on('plotly_buttonclicked', function(ev) {{
  var label = ev.button.label;
  if (label === '▶ Blink') {{
    if (blinkTimer) return;
    blinkTimer = setInterval(function() {{ blinkState = 1 - blinkState; render(); }}, 400);
  }} else if (label === '⏸ Stop') {{
    clearInterval(blinkTimer); blinkTimer = null;
  }}
}});
"""


def sdc_ortho_qc_plot(
    orig_path: str,
    sdc_path: str,
    crosshair: tuple = None,
    pct_clip: float = 99.5,
    title: str = None,
):
    """
    Combines `sdc_qc_plot`'s distorted/corrected/difference/flicker columns
    with `ortho_view_plot`'s click-crosshair: a 3 (sagittal/coronal/axial)
    x 4 (distorted/corrected/diff/flicker) grid, all 3 rows sharing one
    crosshair. Click any panel to move the crosshair — the other two rows
    jump to the matching slice. The flicker column (in every row) blinks
    between distorted and corrected together, via the Blink/Stop buttons,
    so the geometric shift is easy to spot at whichever location you're
    inspecting. Pure Plotly, no server — exportable via `fig.write_html()`.

    *orig_path* and *sdc_path* must be on the same grid (same shape +
    affine), as with `sdc_qc_plot`.

    Parameters
    ----------
    crosshair : (i, j, k), optional
        Initial crosshair voxel indices. Defaults to the volume centre.

    Returns
    -------
    fig, post_script : the Plotly figure and a JS snippet implementing the
        click/blink handlers. Export with::

            fig, post_script = sdc_ortho_qc_plot(orig_sbref, sdc_sbref)
            fig.write_html('sdc_qc.html', div_id='sdc_ortho_qc_plot',
                            post_script=post_script)

        `div_id` must match the module constant `_SDC_ORTHO_DIV_ID` baked
        into `post_script` (the value shown above is correct).

    Note on file size: this bakes 2 x (nx + ny + nz) frames (one per depth
    per blink state), each carrying 4 heatmaps, so the exported HTML is
    noticeably larger than `sdc_qc_plot` or `ortho_view_plot` alone —
    expect tens of MB for a typical sbref-sized volume. Fine for a local
    QC report; downsample first if that becomes a problem.
    """
    orig = _canonical_data(orig_path)
    sdc = _canonical_data(sdc_path)
    if orig.shape != sdc.shape:
        raise ValueError(
            'sdc_ortho_qc_plot: shape mismatch, orig {} vs sdc {}'.format(orig.shape, sdc.shape))

    nx, ny, nz = orig.shape
    i0, j0, k0 = crosshair if crosshair else (nx // 2, ny // 2, nz // 2)
    diff = sdc - orig

    vmax = np.percentile(orig, pct_clip)
    dmax = np.percentile(np.abs(diff), pct_clip) or 1.0
    common = dict(zmin=0, zmax=vmax, colorscale='gray', showscale=False)
    diff_common = dict(zmin=-dmax, zmax=dmax, colorscale='RdBu', showscale=False)

    def slices_sag(d): return orig[d, :, :], sdc[d, :, :], diff[d, :, :]
    def slices_cor(d): return orig[:, d, :], sdc[:, d, :], diff[:, d, :]
    def slices_ax(d): return orig[:, :, d], sdc[:, :, d], diff[:, :, d]

    def row_frames(name, n, slicer, traces):
        fr = []
        for d in range(n):
            o, s, df = slicer(d)
            fr.append(go.Frame(
                name='{}-{}-orig'.format(name, d), traces=traces,
                data=[go.Heatmap(z=o, **common), go.Heatmap(z=s, **common),
                      go.Heatmap(z=df, **diff_common), go.Heatmap(z=o, **common)]))
            fr.append(go.Frame(
                name='{}-{}-sdc'.format(name, d), traces=traces,
                data=[go.Heatmap(z=o, **common), go.Heatmap(z=s, **common),
                      go.Heatmap(z=df, **diff_common), go.Heatmap(z=s, **common)]))
        return fr

    frames = (
        row_frames('sag', nx, slices_sag, [0, 1, 2, 3])
        + row_frames('cor', ny, slices_cor, [4, 5, 6, 7])
        + row_frames('ax', nz, slices_ax, [8, 9, 10, 11])
    )

    subplot_titles = []
    for pname in ('sagittal', 'coronal', 'axial'):
        subplot_titles += ['{}: distorted'.format(pname), '{}: corrected'.format(pname),
                            '{}: diff'.format(pname), '{}: flicker'.format(pname)]

    fig = make_subplots(rows=3, cols=4, subplot_titles=subplot_titles, vertical_spacing=0.08)
    for row, (o, s, df) in enumerate([slices_sag(i0), slices_cor(j0), slices_ax(k0)], start=1):
        fig.add_trace(go.Heatmap(z=o, **common), row=row, col=1)
        fig.add_trace(go.Heatmap(z=s, **common), row=row, col=2)
        fig.add_trace(go.Heatmap(z=df, **diff_common), row=row, col=3)
        fig.add_trace(go.Heatmap(z=o, **common), row=row, col=4)
    fig.frames = frames

    line = dict(color='lime', width=1)
    row_geom = {
        'sag': dict(axis_ids=[1, 2, 3, 4], xval=k0, yval=j0, xspan=nz - 1, yspan=ny - 1),
        'cor': dict(axis_ids=[5, 6, 7, 8], xval=k0, yval=i0, xspan=nz - 1, yspan=nx - 1),
        'ax': dict(axis_ids=[9, 10, 11, 12], xval=j0, yval=i0, xspan=ny - 1, yspan=nx - 1),
    }
    shapes = []
    for name in ('sag', 'cor', 'ax'):
        g = row_geom[name]
        for aid in g['axis_ids']:
            xref = 'x' if aid == 1 else 'x{}'.format(aid)
            yref = 'y' if aid == 1 else 'y{}'.format(aid)
            shapes.append(dict(type='line', xref=xref, yref=yref,
                                x0=g['xval'], x1=g['xval'], y0=0, y1=g['yspan'], line=line))
            shapes.append(dict(type='line', xref=xref, yref=yref,
                                x0=0, x1=g['xspan'], y0=g['yval'], y1=g['yval'], line=line))

    fig.update_layout(
        title=title or 'SDC ortho QC',
        height=750,
        width=1350,
        margin=dict(t=60, b=40, l=20, r=20),
        shapes=shapes,
        updatemenus=[dict(
            type='buttons',
            direction='left',
            x=1.0, y=1.08, xanchor='right', yanchor='top',
            showactive=False,
            buttons=[
                dict(label='▶ Blink', method='skip', args=[]),
                dict(label='⏸ Stop', method='skip', args=[]),
            ],
        )],
    )
    for n in range(1, 13):
        ax_name = 'xaxis' if n == 1 else 'xaxis{}'.format(n)
        fig.update_layout({ax_name: dict(visible=False)})
    for n in range(1, 13):
        ax_name = 'yaxis' if n == 1 else 'yaxis{}'.format(n)
        x_anchor = 'x' if n == 1 else 'x{}'.format(n)
        fig.update_layout({ax_name: dict(visible=False, scaleanchor=x_anchor)})

    post_script = _SDC_ORTHO_CLICK_JS.format(div_id=_SDC_ORTHO_DIV_ID, nx=nx, ny=ny, nz=nz, i0=i0, j0=j0, k0=k0)
    return fig, post_script
