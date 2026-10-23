

from __future__ import annotations
import argparse
import asyncio
import base64
import colorsys
import csv
import hashlib
import io
import json
import logging
import math
import re
import sys
import threading
import zipfile
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize, to_hex, to_rgb
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle, Patch
try:
    import flet as ft
except ImportError:
    ft = None

VERSION = "3.5.0"
FLET_VERSION = "1.0.0"
DAY_SECONDS = 86400
MAX_MATRIX_CELLS = 12_000_000
MAX_IMPORT_BYTES = 256 * 1024 * 1024
MPL_LOCK = threading.RLock()
LOG = logging.getLogger("actogram_Lab")
AUTO = "__auto__"
NONE = "__none__"
TIMESTAMP_COLUMNS = ('bin_start_iso','timestamp_local','timestamp','datetime','date_time',
                     'timestamp_iso','timestamp_utc','iso_time','time')
NAMED_COLOURS = {
    "Ink": "#20252B", "Orange": "#E77722", "Blue": "#2878B5",
    "Teal": "#148A8A", "Green": "#408C4B", "Purple": "#8251B8",
    "Red": "#C84646", "Rose": "#CE5D87", "Gold": "#F2CB62",
    "Brown": "#916747", "Slate": "#667889", "White": "#FFFFFF",
    "Black": "#000000", "Light grey": "#DCE1E5", "Pale red": "#F4B8B8",
}
CMAPS = {
    "viridis": "Viridis", "cividis": "Cividis", "magma": "Magma",
    "inferno": "Inferno", "plasma": "Plasma", "Oranges": "Orange",
    "Blues": "Blue", "Greens": "Green", "Purples": "Purple",
    "Greys": "Greyscale", "group": "White → individual / group colour",
    "custom": "Custom two-colour gradient",
}
STYLES = {
    "dots": "Classic dots", "hticks": "Classic horizontal ticks",
    "ticks": "Classic vertical ticks", "barlength": "Activity-length bars",
    "bars": "Activity-height bars", "heatmap": "Heatmap",
}
STYLE_HINTS = {
    "dots": "One fixed-size dot marks each bin that meets the activity threshold.",
    "hticks": "One fixed-length tick along the time axis marks each bin that meets the activity threshold. Its length does not represent the amount of activity.",
    "ticks": "One fixed-height tick across the day row marks each bin that meets the activity threshold. Its height does not represent the amount of activity.",
    "barlength": "Bar length along the time axis increases with activity in that bin, using the selected scale. ",
    "bars": "Bar height within each day increases with activity in that bin, using the selected scale. ",
    "heatmap": "Each complete bin is coloured by its activity value.",
}
METRICS = {
    "auto": "Automatic · imported activity units",
    "seconds": "Activity seconds / bin", "minutes": "Activity minutes / bin",
    "percent": "Activity / observed time (%)", "episodes": "Episode starts / bin",
    "active_percent": "Individuals meeting the threshold (%)",
}
CYCLES = {"LD": "Light / dark cycle", "DD": "Constant darkness (DD)",
          "LL": "Constant light (LL)", "NONE": "No illumination annotation"}

class DataError(ValueError):
    """An actionable import, plotting or schedule error."""

@dataclass
class ImportOptions:
    timestamp: str = AUTO
    activity: str = AUTO
    subject: str = AUTO
    group: str = AUTO
    duration: str = AUTO
    observed: str = AUTO
    episodes: str = AUTO
    units: str = "seconds"
    fallback_seconds: float = 60.0
    time_mode: str = "wall"
    decimal: str = "."
    day_first: bool = True
    recording_start: str = ""
    recording_end: str = ""
    data_mode: str = "binned"  # binned, samples, events
    aggregation: str = "activity"  # activity, sum, mean
    sample_seconds: float = 1.0
    raw_bin_minutes: float = 1.0
    continuous_events: bool = False
    timestamp_kind: str = "calendar"  # calendar, unix_s, unix_ms, elapsed_s
    value_unit: str = "Activity units"
    date_column: str = NONE
    date_order: str = ""

@dataclass
class Treatment:
    name: str = "Baseline"
    start: str = ""
    end: str = ""
    cycle: str = "LD"
    lights_on: str = "06:00"
    lights_off: str = "18:00"
    light_name: str = "White light"
    light_lux: float | None = None
    dark_name: str = "Dark"
    dark_lux: float | None = 0.0
    light_colour: str = "#F2CB62"
    dark_colour: str = "#303944"

    def validate(self, require_dates: bool = True) -> None:
        if not self.name.strip():
            raise DataError("Give each treatment period a name.")
        if self.cycle not in CYCLES:
            raise DataError(f"{self.name}: choose LD, DD, LL or no annotation.")
        if require_dates:
            if not self.start or not self.end:
                raise DataError(f"{self.name}: enter both the start AND end date/time.")
            if parse_datetime(self.start) >= parse_datetime(self.end):
                raise DataError(f"{self.name}: end must be after start.")
        if self.cycle == "LD" and parse_clock(self.lights_on) == parse_clock(self.lights_off):
            raise DataError(f"{self.name}: lights-on and lights-off cannot be identical. Choose DD or LL instead.")
        if self.cycle == "LD":
            parse_clock(self.lights_on); parse_clock(self.lights_off)
        for value in (self.light_lux, self.dark_lux):
            if value is not None and (not math.isfinite(float(value)) or float(value) < 0):
                raise DataError(f"{self.name}: lux must be nonnegative or left blank.")
        for colour in (self.light_colour, self.dark_colour):
            try:
                to_rgb(colour)
            except ValueError as exc:
                raise DataError(f"{self.name}: invalid phase colour.") from exc

@dataclass
class Settings:
    style: str = "hticks"
    bin_minutes: float = 1.0
    metric: str = "auto"
    orientation: str = "days_down"
    double_plot: bool = True
    alignment: str = "clock24"
    day_start: str = "00:00"
    window_start: str = ""
    window_end: str = ""
    trim_first_hours: float = 0.0
    trim_last_hours: float = 0.0
    excluded_dates: str = ""
    crop_policy: str = "complete"
    bin_policy: str = "start"
    timezone_offset_min: int = 0
    threshold: float = 0.0
    comparison: str = "ge"
    minimum_coverage: float = 100.0
    group_active_percent: float = 50.0
    minimum_subjects: int = 1
    schedule_enabled: bool = True
    schedule_mode: str = "simple"
    simple_cycle: str = "LD"
    lights_on: str = "06:00"
    lights_off: str = "18:00"
    light_colour: str = "#F2CB62"
    dark_colour: str = "#303944"
    treatments: list[dict[str, Any]] = field(default_factory=list)
    shade_mode: str = "both"
    shade_opacity: float = 0.18
    heatmap_schedule: str = "shade"
    schedule_strip: float = 0.16
    show_schedule_legend: bool = True
    cmap: str = "viridis"
    reverse_cmap: bool = False
    low_colour: str = "#FFFFFF"
    high_colour: str = "#E77722"
    missing_colour: str = "#DCE1E5"
    background_colour: str = "#FFFFFF"
    ink_colour: str = "#20252B"
    use_group_colours: bool = True
    mark_colour_mode: str = "solid"
    scale_mode: str = "shared"
    vmax: float | None = None
    show_colourbar: bool = True
    point_size: float = 2.5
    line_width: float = 0.85
    mark_length: float = 0.8
    bar_scale: str = "global"
    figure_width: float = 11.4
    row_height: float = 0.18
    font_size: float = 10.0
    tick_hours: float = 6.0
    tick_days: int = 0
    tick_length: float = 3.5
    tick_width: float = 0.8
    show_grid: bool = False
    label_mode: str = "dates"
    export_dpi: int = 300
    title: str = ""
    x_axis_title: str = ""
    y_axis_title: str = ""
    solar_enabled: bool = False
    solar_city: str = ""
    solar_latitude: float = 69.6492
    solar_longitude: float = 18.9553
    solar_timezone: str = "Europe/Oslo"
    solar_time_basis: str = "local"
    solar_horizon: bool = True
    solar_civil: bool = True
    solar_nautical: bool = False
    solar_astronomical: bool = False
    solar_horizon_colour: str = "#D5A000"
    solar_civil_colour: str = "#1A9FD0"
    solar_nautical_colour: str = "#B44ED4"
    solar_astronomical_colour: str = "#E77722"

    @property
    def bin_seconds(self) -> int:
        return int(round(float(self.bin_minutes) * 60))

    @property
    def start_seconds(self) -> int:
        return parse_clock(self.day_start) if self.alignment == "clock24" else 0

    def validate(self) -> None:
        if self.heatmap_schedule in {"strip", "lines"}:
            self.heatmap_schedule = "shade"
        b = float(self.bin_minutes) * 60
        if not math.isfinite(b) or b < 1 or abs(b-round(b)) > 1e-7 or DAY_SECONDS % int(round(b)):
            raise DataError("Display bins must divide 24 h into whole seconds: e.g. 1, 5, 10, 15, 30 or 60 minutes.")
        allowed = {
            "style": STYLES, "metric": METRICS, "orientation": {"days_down", "days_across"},
            "alignment": {"clock24", "start24"}, "crop_policy": {"complete", "start"},
            "bin_policy": {"start", "strict"}, "comparison": {"ge", "gt"},
            "schedule_mode": {"simple", "multiple"}, "simple_cycle": CYCLES,
            "shade_mode": {"both", "light", "dark", "none"},
            "heatmap_schedule": {"shade", "none"}, "cmap": CMAPS,
            "mark_colour_mode": {"solid", "intensity"}, "scale_mode": {"shared", "auto", "manual"},
            "bar_scale": {"global", "per_day"}, "label_mode": {"dates", "days"},
        }
        for key, values in allowed.items():
            if getattr(self,key) not in values:
                raise DataError(f"Invalid {key.replace('_',' ')} selection.")
        parse_clock(self.day_start)
        for t in (self.window_start, self.window_end):
            if t: parse_datetime(t)
        if self.window_start and self.window_end and parse_datetime(self.window_start) >= parse_datetime(self.window_end):
            raise DataError("The plotting end date/time must be after the start date/time.")
        parse_excluded_dates(self.excluded_dates)
        ranges = {
            "threshold": (0,1e15), "minimum_coverage": (0,100), "group_active_percent": (0,100),
            "trim_first_hours": (0,1e6), "trim_last_hours": (0,1e6), "timezone_offset_min": (-1440,1440),
            "minimum_subjects": (1,100000), "point_size": (.2,12), "line_width": (.1,8),
            "mark_length": (.05,1), "figure_width": (5,30), "row_height": (.06,1.2),
            "font_size": (6,24), "tick_hours": (.5,24), "tick_days": (0,3650),
            "tick_length": (0,15), "tick_width": (.1,5), "shade_opacity": (0,1),
            "schedule_strip": (.04,.35), "export_dpi": (72,600),
        }
        for key,(lo,hi) in ranges.items():
            v=float(getattr(self,key))
            if not math.isfinite(v) or not lo <= v <= hi:
                raise DataError(f"{key.replace('_',' ').capitalize()} must be between {lo:g} and {hi:g}.")
        if self.scale_mode == "manual" and (self.vmax is None or not math.isfinite(self.vmax) or self.vmax <= 0):
            raise DataError("Enter an intensity maximum greater than zero, or choose an automatic scale.")
        for key in ("low_colour","high_colour","missing_colour","background_colour","ink_colour","light_colour","dark_colour"):
            try: to_rgb(getattr(self,key))
            except ValueError as exc: raise DataError(f"Invalid {key.replace('_',' ')}.") from exc
        if self.solar_enabled:
            if self.alignment != 'clock24':
                raise DataError('Solar curves require actual clock-time alignment, not elapsed recording days.')
            if not (-90 <= float(self.solar_latitude) <= 90 and -180 <= float(self.solar_longitude) <= 180):
                raise DataError('Solar latitude must be -90 to 90 and longitude -180 to 180.')
            try: ZoneInfo(self.solar_timezone)
            except Exception as exc: raise DataError('Enter a valid IANA time zone, e.g. Europe/Oslo.') from exc
            if self.solar_time_basis not in {'local','standard','utc'}:
                raise DataError('Choose local time, fixed standard time or UTC for solar curves.')
            for k in ('solar_horizon_colour','solar_civil_colour','solar_nautical_colour','solar_astronomical_colour'):
                try: to_rgb(getattr(self,k))
                except ValueError as exc: raise DataError(f'Invalid solar curve colour: {k}.') from exc
        if self.schedule_enabled:
            validated_treatments(self)

@dataclass
class Subject:
    id: str
    group: str
    data: pd.DataFrame
    label: str = ""
    colour: str | None = None
    included: bool = True
    sources: list[dict[str, Any]] = field(default_factory=list)
    measure_kind: str = "duration"
    value_unit: str = "Activity seconds"
    aggregation: str = "sum"
    @property
    def name(self) -> str:
        return self.label or self.id

@dataclass
class Panel:
    key: str
    name: str
    group: str
    is_group: bool
    colour: str
    seconds: np.ndarray
    observed: np.ndarray
    episodes: np.ndarray
    values: np.ndarray
    binary: np.ndarray
    active_percent: np.ndarray
    n: np.ndarray
    n_binary: np.ndarray
    members: int
    origin: pd.Timestamp | None = None
    measure_kind: str = "duration"
    unit: str = "Activity seconds / bin"
    unfiltered_values: np.ndarray | None = None
    threshold_units: str = ""

@dataclass
class Analysis:
    settings: Settings
    dates: pd.DatetimeIndex
    shown_days: int
    panels: dict[str, Panel]
    shared_vmax: dict[bool,float]
    notes: list[str]
    row_indices: np.ndarray = field(default_factory=lambda: np.array([],dtype=int))
    base_day: int = 0

def parse_clock(value: str, allow_24: bool = False) -> int:
    value = str(value).strip()
    if not re.fullmatch(r"\d{1,2}:\d{2}(?::\d{2})?", value):
        raise DataError("Times must use HH:MM or HH:MM:SS, for example 06:00.")
    parts = list(map(int, value.split(":")))
    h, m, sec = parts[0], parts[1], parts[2] if len(parts) > 2 else 0
    if allow_24 and h == 24 and m == sec == 0:
        return DAY_SECONDS
    if h > 23 or m > 59 or sec > 59:
        raise DataError("Clock times must be between 00:00:00 and 23:59:59.")
    return h * 3600 + m * 60 + sec

def format_clock(seconds: int, with_seconds: bool = False) -> str:
    seconds = int(seconds) % DAY_SECONDS
    h, rem = divmod(seconds, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}" if with_seconds else f"{h:02d}:{m:02d}"

def snail_id_from_path(path: str | Path) -> str:
    # Same token-based filename extraction as the supplied Tkinter application.
    stem = Path(path).stem
    tokens = re.split(r"[ _\-.]+", stem)
    patterns = (r"[A-Z]\d{2,3}", r"FEDX[A-Z]?\d{1,3}", r"[A-Z]{1,3}\d{1,3}")
    for token in tokens:
        if any(re.fullmatch(p, token, re.I) for p in patterns):
            return token.upper()
    short = re.split(r"_binned(?:_|$)", stem, flags=re.I)[0]
    return short or stem or "snail"

def nominal_source_seconds(subject: Subject) -> float:
    durations = subject.data.duration.to_numpy(float)
    vals, counts = np.unique(np.round(durations, 6), return_counts=True)
    return float(vals[np.argmax(counts)])

def recommended_bin_seconds(subjects: list[Subject], current: int = 60) -> int:
    """Smallest whole-second day divisor that fits all ordinary source bins."""
    selected = [s for s in subjects if s.included]
    if not selected:
        return current
    longest = max(float(s.data.duration.max()) for s in selected)
    nominal = [nominal_source_seconds(s) for s in selected]
    for candidate in range(max(1, int(math.ceil(longest)), int(current)), DAY_SECONDS + 1):
        if DAY_SECONDS % candidate == 0 and all(abs(candidate / n - round(candidate / n)) < 1e-6 for n in nominal):
            return candidate
    return max(current, int(math.ceil(longest)))

def normal_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")

def resolve_column(frame: pd.DataFrame, selected: str, aliases: tuple[str, ...], required: bool = False) -> str | None:
    if selected == NONE:
        if required:
            raise DataError("A required input column cannot be set to None.")
        return None
    lookup = {normal_name(c): c for c in frame.columns}
    if selected and selected != AUTO:
        c = lookup.get(normal_name(selected))
        if c is None:
            raise DataError(f"Column {selected!r} was not found in this file.")
        return c
    for alias in aliases:
        if alias in lookup:
            return lookup[alias]
    if required:
        raise DataError(f"Could not identify the {aliases[0]} column. Select it in the import dialog.")
    return None

@dataclass
class TextImportOptions:
    delimiter: str = 'auto'
    skip_rows: int = 0
    header: bool = True
    column_names: str = ''
    encoding: str = 'auto'
    widths: str = ''
    pad_short_rows: bool = False
    record_type: str = ''


def decode_text_file(raw: bytes, encoding: str = 'auto') -> tuple[str,str]:
    if len(raw)>MAX_IMPORT_BYTES:
        raise DataError('A single input file exceeds the 256 MB import limit.')
    if not raw:
        raise DataError('This file is empty.')
    aliases={'utf8':'utf-8-sig','utf-8':'utf-8-sig','utf16':'utf-16','windows':'cp1252'}
    encoding=aliases.get(encoding,encoding)
    if encoding=='auto':
        if raw.startswith((b'\xff\xfe',b'\xfe\xff')):
            encoding='utf-16'
        elif b'\x00' in raw[:4096]:
            block=raw[:4096]
            even=block[::2].count(0); odd=block[1::2].count(0)
            encoding='utf-16-le' if odd>even else 'utf-16-be'
        else:
            try:
                return raw.decode('utf-8-sig'),'utf-8-sig'
            except UnicodeDecodeError:
                encoding='cp1252'
    if encoding not in {'utf-8-sig','utf-16','utf-16-le','utf-16-be','cp1252'}:
        raise DataError('Choose UTF-8, UTF-16 or Windows-1252 text encoding.')
    try:
        text=raw.decode(encoding)
    except UnicodeError as exc:
        raise DataError(f'Could not decode this file as {encoding}. Choose a different text encoding.') from exc
    if '\x00' in text:
        raise DataError('The decoded file contains zero bytes. Check its text encoding.')
    return text.lstrip('\ufeff'),encoding


def text_preview_lines(raw: bytes, encoding: str = 'auto', limit: int = 12) -> str:
    text,_=decode_text_file(raw,encoding)
    lines=text.splitlines()
    return '\n'.join(f'{i+1:>3}  '+(line[:240]+'…' if len(line)>240 else line)
                     for i,line in enumerate(lines[:max(1,min(int(limit),100))]))


def whitespace_fields(line: str, line_number: int) -> list[str]:
    fields=[]; i=0; length=len(line)
    while i<length:
        while i<length and line[i].isspace(): i+=1
        if i==length: break
        if line[i]=='"':
            i+=1; value=[]; closed=False
            while i<length:
                if line[i]=='"':
                    if i+1<length and line[i+1]=='"':
                        value.append('"'); i+=2
                    else:
                        i+=1; closed=True; break
                else:
                    value.append(line[i]); i+=1
            if not closed:
                raise DataError(f'Line {line_number}: a quoted whitespace field is not closed. Keep each record on one line or choose its CSV/tab delimiter.')
            if i<length and not line[i].isspace():
                raise DataError(f'Line {line_number}: add a space after the closing quote.')
            fields.append(''.join(value))
        else:
            start=i
            while i<length and not line[i].isspace(): i+=1
            fields.append(line[start:i])
    return fields


def detect_text_delimiter(text: str) -> str:
    lines=[line for line in text.splitlines()[:100] if line.strip()]
    if not lines:
        raise DataError('There are no data lines after the skipped lines.')
    sample='\n'.join(lines)
    candidates=[]
    for priority,(name,separator) in enumerate((('tab','\t'),('semicolon',';'),('pipe','|'),('comma',','))):
        try:
            rows=list(csv.reader(io.StringIO(sample),delimiter=separator,strict=True))
        except csv.Error:
            continue
        widths=[len(row) for row in rows if row]
        widths=[width for width in widths if width>1]
        if widths:
            modal=max(set(widths),key=lambda value:(widths.count(value),value))
            consistent=widths.count(modal)
            candidates.append((consistent,len(widths),-priority,modal,name))
    if candidates:
        best=max(candidates)
        if best[-1]=='comma' and best[3]<=2:
            try:
                space_widths=[len(whitespace_fields(line,i+1)) for i,line in enumerate(lines)]
                if space_widths and min(space_widths)>=3 and len(set(space_widths))==1:
                    return 'whitespace'
            except DataError:
                pass
        return best[-1]
    try:
        widths=[len(whitespace_fields(line,i+1)) for i,line in enumerate(lines)]
        if max(widths,default=0)>1: return 'whitespace'
    except DataError:
        return 'whitespace'
    return 'comma'


def text_rows(text: str, delimiter: str, skip_rows: int, widths: str = ''):
    lines=text.splitlines(keepends=True)
    lines=lines[skip_rows:]
    if delimiter=='auto': delimiter=detect_text_delimiter(''.join(lines[:100]))
    if delimiter=='fixed':
        try:
            sizes=[int(value.strip()) for value in str(widths).split(',')]
        except ValueError as exc:
            raise DataError('Enter positive fixed column widths separated by commas, for example 10,9,5.') from exc
        if not sizes or any(value<=0 for value in sizes):
            raise DataError('Enter positive fixed column widths separated by commas, for example 10,9,5.')
        total=sum(sizes)
        for number,line in enumerate(lines,skip_rows+1):
            line=line.rstrip('\r\n')
            if not line.strip():
                yield number,[]
                continue
            if line[total:].strip():
                raise DataError(f'Line {number}: text remains after the fixed-width columns. Increase the widths or add another column.')
            row=[]; start=0
            for size in sizes:
                row.append(line[start:start+size].strip()); start+=size
            yield number,row
        return
    if delimiter=='whitespace':
        for number,line in enumerate(lines,skip_rows+1):
            yield number,whitespace_fields(line.rstrip('\r\n'),number)
        return
    separator={'comma':',','tab':'\t','semicolon':';','pipe':'|'}.get(delimiter,delimiter)
    if len(separator)!=1 or separator in {'\r','\n','"'}:
        raise DataError('Choose a separator or enter one character for a custom separator.')
    reader=csv.reader(iter(lines),delimiter=separator,strict=True)
    previous=0
    try:
        for row in reader:
            number=skip_rows+previous+1; previous=reader.line_num
            yield number,[value.strip() for value in row]
    except csv.Error as exc:
        raise DataError(f'Line {skip_rows+reader.line_num}: malformed quoted text. Check the separator and quotation marks.') from exc


def text_column_names(value: str) -> list[str]:
    if not str(value or '').strip(): return []
    try:
        rows=list(csv.reader(io.StringIO(str(value)),strict=True))
    except csv.Error as exc:
        raise DataError('Enter column names separated by commas.') from exc
    if len(rows)!=1:
        raise DataError('Enter column names on one line, separated by commas.')
    return [name.strip() for name in rows[0]]


def text_table(raw: bytes, options: TextImportOptions | None = None) -> pd.DataFrame:
    options=options or TextImportOptions()
    text,encoding=decode_text_file(raw,options.encoding)
    try:
        skip=int(options.skip_rows)
        if float(options.skip_rows)!=skip or skip<0: raise ValueError
    except (ValueError,TypeError,OverflowError) as exc:
        raise DataError('The number of lines to skip must be a nonnegative whole number.') from exc
    if skip>=len(text.splitlines()):
        raise DataError('There are no data lines after the skipped lines.')
    delimiter=options.delimiter
    if delimiter=='auto': delimiter=detect_text_delimiter('\n'.join(text.splitlines()[skip:skip+100]))
    rows=[]; header=None; blank=0; source_count=0; filtered=0
    selected=str(options.record_type or '').strip()
    for number,row in text_rows(text,delimiter,skip,options.widths):
        if not row:
            blank+=1
            continue
        if options.header and header is None:
            header=[value.strip() for value in row]
            continue
        source_count+=1
        if selected and (not row or row[0].strip()!=selected):
            filtered+=1
            continue
        rows.append((number,row))
    if not rows:
        if selected: raise DataError(f'No data rows have {selected!r} in the first column. Check the record filter, skipped lines and header setting.')
        raise DataError('This selection has no data rows. Check skipped lines and whether the first row is a header.')
    names=text_column_names(options.column_names)
    manual_names=bool(names)
    if not names and header is not None: names=header
    expected=len(names) if names else len(rows[0][1])
    if options.pad_short_rows: expected=max(expected,max(len(row) for _,row in rows))
    if manual_names and (any(not name for name in names) or len({normal_name(name) for name in names})!=len(names)):
        raise DataError('Replacement column names must be nonempty and distinct.')
    added=[]; renamed=[]
    for index in range(len(names),expected):
        name=f'Column{index+1}'
        names.append(name); added.append(name)
    reserved={normal_name(name) for name in names if name}
    used=set()
    for index,name in enumerate(names):
        original=name
        if not normal_name(name): name=f'Column{index+1}'
        if normal_name(name) in used:
            count=2; base=name
            while normal_name(f'{base}_{count}') in used|reserved: count+=1
            name=f'{base}_{count}'
        if name!=original:
            renamed.append({'column':index+1,'from':original,'to':name})
            names[index]=name
        used.add(normal_name(name))
    values=[]; padded=0
    for number,row in rows:
        if len(row)<expected and options.pad_short_rows:
            row=row+['']*(expected-len(row)); padded+=1
        elif len(row)!=expected:
            hint=' Enable padding if shorter records intentionally omit trailing fields.' if len(row)<expected else ''
            raise DataError(f'Line {number}: found {len(row)} columns; expected {expected}. Check the separator, header setting or skipped lines.{hint}')
        values.append(row)
    frame=pd.DataFrame(values,columns=names,dtype=str)
    frame.attrs.update(delimiter=delimiter,encoding=encoding,skipped_lines=skip,source_rows=source_count,
                       converted_rows=len(values),filtered_rows=filtered,padded_rows=padded,blank_lines=blank,
                       renamed_columns=renamed,added_columns=added)
    return frame


def suggest_text_options(raw: bytes) -> TextImportOptions:
    text,encoding=decode_text_file(raw)
    delimiter=detect_text_delimiter(text)
    options=TextImportOptions(delimiter=delimiter,encoding=encoding)
    try:
        first=next(row for _,row in text_rows(text,delimiter,0) if row)
    except (StopIteration,DataError):
        return options
    data_pattern=(r'^(?:[-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?|'
                  r'\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}(?:[Tt\s].*)?|'
                  r'\d{1,2}:\d{2}(?::\d{2}(?:[.,]\d+)?)?(?:\s*[APap][Mm])?)$')
    data=sum(bool(re.match(data_pattern,value.strip())) for value in first)
    options.header=data<len(first)/2
    if delimiter=='tab' and len(first)>=3 and [value.strip().lower() for value in first[:3]]==['f','date','hmsec']:
        records=[]
        for _,row in text_rows('\n'.join(text.splitlines()[:100]),delimiter,0):
            if row: records.append(row)
        if len(records)>1 and all(row[0].strip() in {'D','T','S'} for row in records[1:]):
            options.header=True; options.pad_short_rows=True
    return options


def read_table(raw: bytes) -> pd.DataFrame:
    if len(raw) > MAX_IMPORT_BYTES:
        raise DataError("A single input file exceeds the 256 MB import limit.")
    text,_ = decode_text_file(raw)
    try:
        sep = csv.Sniffer().sniff(text[:65536], delimiters=",;\t|").delimiter
    except csv.Error:
        sep = ","
    try:
        result = pd.read_csv(io.StringIO(text), sep=sep, dtype=str)
    except Exception as exc:
        raise DataError(f"Could not read the delimited text file: {exc}") from exc
    if not isinstance(result.index,pd.RangeIndex):
        raise DataError('Some rows contain more fields than the column headings. Use Convert text to CSV to preview and retain all fields.')
    result.columns = [str(c).strip() for c in result.columns]
    if result.empty:
        raise DataError("This file has no data rows.")
    if len({normal_name(c) for c in result.columns}) != len(result.columns):
        raise DataError("The file contains duplicate or ambiguous column names.")
    return result

def numeric(series: pd.Series, decimal: str = ".") -> np.ndarray:
    s = series.astype(str).str.strip()
    if decimal == ",":
        s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)

def automatic_date_order(series: pd.Series) -> str:
    text=series.astype(str).str.strip()
    parts=text.str.extract(r'^(\d{1,4})([-/.])(\d{1,2})\2(\d{1,4})(?:[Tt\s]|$)')
    year_first=parts[0].str.len().eq(4)&parts[3].str.len().le(2)
    year_last=parts[3].str.len().eq(4)&parts[0].str.len().le(2)
    invalid=parts[0].isna()|~(year_first|year_last)
    if invalid.any():
        examples=', '.join(repr(v) for v in text.loc[invalid].drop_duplicates().head(2))
        raise DataError(f'Automatic date order needs numeric dates with a four-digit year. Check these file values: {examples}.')
    if year_first.any() and year_last.any():
        raise DataError('The timestamp column mixes year-first and year-last dates. Prepare a consistent date format before importing.')
    a=pd.to_numeric(parts[2] if year_first.all() else parts[0]).to_numpy(int)
    b=pd.to_numeric(parts[3] if year_first.all() else parts[2]).to_numpy(int)
    month_first=bool(np.all((a>=1)&(a<=12)&(b>=1)&(b<=31)))
    day_first=bool(np.all((b>=1)&(b<=12)&(a>=1)&(a<=31)))
    if year_first.all():
        if month_first: return 'ymd'
        if day_first: return 'ydm'
    else:
        if month_first and day_first and np.any(a!=b):
            example=text.loc[a!=b].iloc[0]
            raise DataError(f'Automatic cannot distinguish month/day/year from day/month/year in {example!r}. Choose the date order explicitly')
        if day_first: return 'dmy'
        if month_first: return 'mdy'
    raise DataError('The dates do not use one consistent numeric order. Check the month/day values or select the correct date order explicitly.')


def parse_times(series: pd.Series, mode: str, day_first: bool = True, date_order: str = "") -> pd.Series:
    s = series.astype(str).str.strip()
    if s.str.fullmatch(r"[-+]?\d+(\.\d+)?").any():
        raise DataError("Supply an ISO date/time column.")
    if date_order == "auto": date_order = automatic_date_order(s)
    orders = {"dmy": (3, 2, 0, "day / month / year", "20/10/2026"),
              "mdy": (3, 0, 2, "month / day / year", "10/20/2026"),
              "ymd": (0, 2, 3, "year / month / day", "2026-10-20"),
              "ydm": (0, 3, 2, "year / day / month", "2026-20-10")}
    if date_order:
        if date_order not in orders:
            raise DataError("Choose a supported date order: day/month/year, month/day/year, year/month/day or year/day/month.")
        year_index, month_index, day_index, label, example = orders[date_order]
        pattern = (r"^(\d{4})([-/.])(\d{1,2})\2(\d{1,2})((?:[Tt]|\s+).*)?$"
                   if date_order.startswith("y") else
                   r"^(\d{1,2})([-/.])(\d{1,2})\2(\d{4})((?:[Tt]|\s+).*)?$")
        parts = s.str.extract(pattern)
        invalid = parts[0].isna()
        if invalid.any():
            examples = ", ".join(repr(v) for v in s.loc[invalid].drop_duplicates().head(2))
            raise DataError(f"The selected date order is {label}, for example {example} 09:56:00. Check these timestamps: {examples}.")
        tail = parts[4].fillna("").str.strip().str.replace(r"^[Tt]\s*", "", regex=True)
        s = (parts[year_index] + "-" + parts[month_index].str.zfill(2) + "-" + parts[day_index].str.zfill(2)
             + tail.where(tail.eq(""), "T" + tail))
        if mode != "utc":
            s = s.where(tail.eq(""), s.str.replace(r"(?:[Zz]|[+-]\d{2}(?::?\d{2})?)$", "", regex=True).str.strip())
        result = pd.to_datetime(s, format="ISO8601", utc=mode == "utc", errors="coerce")
        if mode == "utc": result = result.dt.tz_localize(None)
        if result.isna().any():
            examples = ", ".join(repr(v) for v in series.loc[result.isna()].drop_duplicates().head(2))
            raise DataError(f"Invalid date or time for {label}: {examples}. Check the date order and use a clock time such as 09:56:00.")
        return result
    try:
        if mode == "utc":
            return pd.to_datetime(s, format="mixed", dayfirst=day_first, utc=True, errors="coerce").dt.tz_localize(None)
        # Fixed 24-hour laboratory/wall-clock plotting. 
        s = s.str.replace(r"(?:Z|[+-]\d{2}:?\d{2})$", "", regex=True)
        return pd.to_datetime(s, format="mixed", dayfirst=day_first, errors="coerce")
    except Exception as exc:
        raise DataError("Could not parse timestamps. Use ISO dates/times and a consistent clock basis.") from exc

def threshold_activity(seconds: np.ndarray, threshold: float, comparison: str) -> np.ndarray:
    # Threshold zero means ANY positive activity, even for the >= comparison.
    hit = (seconds > 0) & ((seconds >= threshold) if comparison == "ge" else (seconds > threshold))
    return np.where(np.isfinite(seconds), hit.astype(float), np.nan)

def finite_mean(array: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    n = np.isfinite(array).sum(axis=axis)
    mean = np.divide(np.nansum(array, axis=axis), n, out=np.full(n.shape, np.nan, dtype=float), where=n > 0)
    return mean, n

def displayed(matrix: np.ndarray, shown: int, double: bool, row_indices: np.ndarray | None = None) -> np.ndarray:
    rows = np.arange(shown) if row_indices is None else row_indices
    
    return np.concatenate([matrix[rows], matrix[rows + 1]], axis=1) if double else matrix[rows].copy()

def safe_filename(value: str) -> str:
    value = re.sub(r"[^\w. -]+", "_", value, flags=re.UNICODE).strip(" ._")
    return (value[:100] or "actogram")

def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, na_rep="").encode("utf-8-sig")


def parse_datetime(text: str) -> pd.Timestamp:
    text = str(text or '').strip()
    if not text:
        raise DataError('Enter a date and time, e.g. 22/09/2026 06:00.')
    if not re.search(r'\d{1,2}:\d{2}', text):
        raise DataError(f'{text!r}: include a time as well as a date, e.g. 22/09/2026 06:00.')
    value = parse_times(pd.Series([text]), 'wall', True).iloc[0]
    if pd.isna(value):
        raise DataError(f'Could not read date/time {text!r}. Use DD/MM/YYYY HH:MM or YYYY-MM-DD HH:MM:SS.')
    return pd.Timestamp(value)


def datetime_text(t: pd.Timestamp) -> str:
    return pd.Timestamp(t).strftime('%d/%m/%Y %H:%M:%S')


def parse_excluded_dates(text: str) -> set[pd.Timestamp]:
    result = set()
    for token in re.split(r'[,;\n]+', str(text).strip()):
        token = token.strip()
        if not token:
            continue
        try:
            t = pd.to_datetime(token, dayfirst=not bool(re.match(r'^\d{4}-',token)), errors='raise')
        except Exception as exc:
            raise DataError(f'Cannot read excluded date {token!r}; separate dates with commas.') from exc
        result.add(pd.Timestamp(t).normalize())
    return result


def validated_treatments(cfg: Settings) -> list[Treatment]:
    if cfg.schedule_mode == 'simple':
        t = Treatment(name='Baseline', cycle=cfg.simple_cycle, lights_on=cfg.lights_on,
                      lights_off=cfg.lights_off, light_colour=cfg.light_colour, dark_colour=cfg.dark_colour)
        t.validate(False)
        return [t]
    if not cfg.treatments:
        raise DataError('Multiple-period mode is selected but there are no treatment periods. Open Advanced treatments.')
    out = []
    for row in cfg.treatments:
        t = Treatment(**{k:v for k,v in row.items() if k in {f.name for f in fields(Treatment)}})
        t.validate(True)
        out.append(t)
    out.sort(key=lambda t: parse_datetime(t.start))
    names = [t.name.strip() for t in out]
    if len(set(names)) != len(names):
        raise DataError('Give each treatment period a distinct name, e.g. Red 50 lux and Red 20 lux.')
    for a,b in zip(out[:-1],out[1:]):
        if parse_datetime(a.end) > parse_datetime(b.start):
            raise DataError(f'Treatment periods overlap: {a.name!r} and {b.name!r}. Ends are exclusive; adjacent periods may share a boundary.')
    return out


def validate_frame(frame: pd.DataFrame, subject_id: str, kind: str = 'duration') -> tuple[pd.DataFrame,int]:
    frame = frame.copy()
    required = {'time','duration','observed','seconds','episodes'}
    if not required.issubset(frame):
        raise DataError(f'{subject_id}: normalised data columns are incomplete.')
    if frame.empty or frame.time.isna().any():
        raise DataError(f'{subject_id}: no rows or invalid timestamps. No rows were silently discarded.')
    if getattr(frame.time.dt,'tz',None) is not None:
        raise DataError('Use one consistent timezone-naive laboratory clock, or import offsets as UTC.')
    if 'value' not in frame:
        frame['value'] = frame.seconds
    if 'weight' not in frame:
        frame['weight'] = np.where(np.isfinite(frame.value),1.,0.)
    if 'elapsed' not in frame:
        frame['elapsed'] = (frame.time-frame.time.min()).dt.total_seconds()
    frame = frame[['time','duration','observed','seconds','episodes','value','weight','elapsed']]
    d,o,s,e,v,w = (frame[k].to_numpy(float) for k in ('duration','observed','seconds','episodes','value','weight'))
    if np.any(~np.isfinite(d) | (d<=0)):
        raise DataError(f'{subject_id}: source intervals must have finite positive durations.')
    if np.any(~np.isfinite(o) | (o<0) | (o>d+1e-6)):
        raise DataError(f'{subject_id}: observed seconds must be between 0 and the source interval duration.')
    if kind == 'duration' and np.any(np.isinf(s) | (np.isfinite(s) & ((s<0)|(s>o+1e-6)))):
        raise DataError(f'{subject_id}: activity exceeds observed time or is negative. Check activity units and source bin duration.')
    if np.any(np.isinf(v) | (np.isfinite(v)&(v<0))):
        raise DataError(f'{subject_id}: activity values must be nonnegative; infinite values are invalid.')
    if np.any(np.isinf(e) | (np.isfinite(e)&((e<0)|(np.abs(e-np.round(e))>1e-6)))):
        raise DataError(f'{subject_id}: episode counts must be nonnegative whole numbers.')
    if np.any(~np.isfinite(w)|(w<0)):
        raise DataError(f'{subject_id}: invalid numeric-mean sample weights.')
    if np.any(~np.isfinite(frame.elapsed.to_numpy(float))):
        raise DataError(f'{subject_id}: invalid elapsed timestamps.')
    unknown = ~np.isfinite(v)|(o<=0)
    frame.loc[unknown,'observed'] = 0.
    frame.loc[unknown,'weight'] = 0.
    frame.loc[unknown,['value','seconds','episodes']] = np.nan
    before = len(frame)
    frame = frame.drop_duplicates(subset=['time','duration','observed','seconds','episodes','value','weight'])
    frame = frame.sort_values('time',kind='stable').reset_index(drop=True)
    starts = frame.time.to_numpy(dtype='datetime64[ns]').astype(np.int64)
    ns = np.rint(frame.duration.to_numpy(float)*1e9).astype(np.int64)
    if np.any(starts[1:] < starts[:-1]+ns[:-1]-1000):
        raise DataError(f'{subject_id}: non-identical intervals overlap. Check the source interval/sample duration, duplicated exports, IDs or repeated daylight-saving clock times.')
    return frame, before-len(frame)


def activity_values(series: pd.Series, opts: ImportOptions, column: str) -> np.ndarray:
    text = series.astype(str).str.strip()
    missing = series.isna() | text.str.lower().isin({'', 'nan', 'na', 'n/a', 'none', 'null'})
    if opts.units in {'binary', 'binary_flag'}:
        mapped = text.str.lower().replace({'active': '1', 'inactive': '0', 'true': '1',
                                          'false': '0', 'yes': '1', 'no': '0', 'on': '1', 'off': '0'})
        values = numeric(mapped, opts.decimal)
        invalid = (~missing.to_numpy()) & (~np.isfinite(values) | ~np.isin(values, [0, 1]))
        if invalid.any():
            examples = ', '.join(repr(v) for v in text.loc[invalid].drop_duplicates().head(3))
            raise DataError(f'{column}: use 0/1 or active/inactive flags. Unrecognised values: {examples}.')
    else:
        values = numeric(text, opts.decimal)
        invalid = (~missing.to_numpy()) & np.isnan(values)
        if invalid.any():
            examples = ', '.join(repr(v) for v in text.loc[invalid].drop_duplicates().head(3))
            raise DataError(f'{column}: activity must be numeric. Check the selected column and decimal separator. Unrecognised values: {examples}.')
    values[missing.to_numpy()] = np.nan
    return values



def source_times(df: pd.DataFrame, opts: ImportOptions) -> tuple[pd.Series,list[str]]:
    aliases=TIMESTAMP_COLUMNS
    if opts.timestamp_kind in {'elapsed_s','elapsed_min'}:
        aliases+=('elapsed_seconds','elapsed_minutes','elapsed','bin_start_sec','t_since_start_s')
    tc = resolve_column(df,opts.timestamp,aliases,False)
    ec = resolve_column(df,AUTO,('bin_start_sec','elapsed_seconds','t_since_start_s'))
    notes = []
    if not tc:
        if ec and opts.recording_start:
            values = numeric(df[ec],opts.decimal)
            if np.any(~np.isfinite(values) | (values < 0)):
                raise DataError('Elapsed timestamps must be nonnegative numbers in seconds.')
            try:
                times = pd.Series(parse_datetime(opts.recording_start)+pd.to_timedelta(values,unit='s'))
            except (ValueError, OverflowError) as exc:
                raise DataError('Elapsed timestamps are outside the supported date range.') from exc
        else:
            raise DataError('Choose the timestamp column. Elapsed timestamps also need a recording start date and time.')
    elif opts.timestamp_kind in {'unix_s','unix_ms','elapsed_s','elapsed_min'}:
        values = numeric(df[tc],opts.decimal)
        if np.any(~np.isfinite(values)):
            raise DataError(f'{tc}: numeric timestamps must contain a valid number in every row.')
        if opts.timestamp_kind in {'unix_s','unix_ms'}:
            unit = 's' if opts.timestamp_kind=='unix_s' else 'ms'
            times = pd.to_datetime(pd.Series(values),unit=unit,utc=True,errors='coerce').dt.tz_localize(None)
            notes.append('Unix timestamps use UTC. Enter schedules and plotting limits in UTC for this import.')
        else:
            if not opts.recording_start:
                raise DataError('Elapsed timestamps need the recording start date and time.')
            if np.any(values < 0):
                raise DataError('Elapsed timestamps must be zero or positive.')
            unit = 's' if opts.timestamp_kind=='elapsed_s' else 'min'
            try:
                times = pd.Series(parse_datetime(opts.recording_start)+pd.to_timedelta(values,unit=unit))
            except (ValueError, OverflowError) as exc:
                raise DataError('Elapsed timestamps are outside the supported date range.') from exc
    else:
        dc = resolve_column(df,opts.date_column,(),False)
        if dc == tc:
            raise DataError('Choose different columns for the date and the time.')
        raw = df[dc].astype(str).str.strip()+' '+df[tc].astype(str).str.strip() if dc else df[tc]
        text = raw.astype(str).str.strip()
        if not dc and text.str.fullmatch(r'\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:\s*[APap][Mm])?(?:Z|[+-]\d{2}:?\d{2})?').any():
            raise DataError('These timestamps contain only a clock time. Select a separate date column, or supply a combined date/time column.')
        times = parse_times(raw,opts.time_mode,opts.day_first,opts.date_order)
        if opts.time_mode=='wall' and text.str.contains(r'(?:Z|[+-]\d{2}:?\d{2})$',regex=True).any():
            notes.append('Using the clock time written in each row; timezone suffixes are ignored.')
    if times.isna().any():
        raise DataError(f'{int(times.isna().sum())} invalid timestamps. Fix them before import; no rows were discarded.')
    return times, notes


def sample_clock(times: pd.Series, elapsed: np.ndarray, text: pd.Series, sid: str) -> tuple[np.ndarray,dict]:
    if np.any(~np.isfinite(elapsed) | (elapsed < 0)):
        raise DataError(f'{sid}: t_since_start_s must contain a nonnegative number in every row.')
    if len(elapsed)>1 and np.any(np.diff(elapsed)<=0):
        raise DataError(f'{sid}: elapsed sample times must increase. Repeated samples or a restarted logger must be imported as separate recordings.')
    try:
        deltas=pd.to_timedelta(elapsed,unit='s')
        offsets=(times.reset_index(drop=True)-deltas).to_numpy(dtype='datetime64[ns]').astype(np.int64)
    except (ValueError,OverflowError) as exc:
        raise DataError(f'{sid}: elapsed sample times are outside the supported date range.') from exc
    digits=text.astype(str).str.extract(r'\d{1,2}:\d{2}:\d{2}[.,](\d+)',expand=False).fillna('').str.len().to_numpy(int)
    precision=np.power(10,9-np.minimum(digits,9)).astype(np.int64)
    
    origin=int(np.median(offsets.astype(np.float64)))
    reconstructed=origin+deltas.to_numpy(dtype='timedelta64[ns]').astype(np.int64)
    residual=np.abs((reconstructed-times.to_numpy(dtype='datetime64[ns]').astype(np.int64))/1e9)
   
    tolerance=np.maximum(precision.astype(float)/1e9+0.25,0.25)
    bad=residual>tolerance
    if np.any(bad):
        worst=int(np.argmax(residual))
        raise DataError(f'{sid}: calendar and elapsed clocks diverge by {residual[worst]:.3f} s '
                        f'at row {worst+1}. This exceeds timestamp precision and normal logger jitter. '
                        'Check for clock resets, incorrect date order or concatenated sessions.')
    ns=reconstructed
    if len(ns)>1 and np.any(np.diff(ns)<=0):
        raise DataError(f'{sid}: elapsed samples are too close to distinguish at nanosecond precision.')
    return ns,{'calendar_quantization':'rounded/whole-second with jitter tolerance',
               'origin_estimate':pd.Timestamp(origin).isoformat(),
               'maximum_clock_residual_seconds':float(residual.max()),
               'clock_tolerance_seconds':float(tolerance.max())}


def sample_interval_totals(starts: np.ndarray, ends: np.ndarray, values: np.ndarray,
                           origin: int, width: int, count: int) -> np.ndarray:
    left=(starts-origin)//width; right=(ends-origin-1)//width
    same=left==right
    total=np.bincount(left[same],weights=(ends[same]-starts[same])*values[same],minlength=count).astype(float)
    cross=~same
    if np.any(cross):
        first=left[cross]; last=right[cross]; weights=values[cross]
        total+=np.bincount(first,weights=(origin+(first+1)*width-starts[cross])*weights,minlength=count)
        total+=np.bincount(last,weights=(ends[cross]-origin-last*width)*weights,minlength=count)
        interior=last>first+1
        if np.any(interior):
            changes=(np.bincount(first[interior]+1,weights=weights[interior],minlength=count+1)
                     -np.bincount(last[interior],weights=weights[interior],minlength=count+1))
            total+=np.cumsum(changes[:-1])*width
    return total/1e9


def decode_tick_samples(df: pd.DataFrame, times: pd.Series, ids: pd.Series, groups: pd.Series,
                        opts: ImportOptions, provenance: dict, ac: str, tc: str, elapsed_column: str,
                        notes: list[str]) -> tuple[list[Subject],list[str]]:
    nominal=float(opts.sample_seconds); b=float(opts.raw_bin_minutes)*60
    if not math.isfinite(nominal) or nominal<=0:
        raise DataError('Enter a positive normal sample interval in seconds.')
    if not math.isfinite(b) or b<1 or abs(b-round(b))>1e-7 or DAY_SECONDS%int(round(b)):
        raise DataError('Raw sample bin size must divide 24 hours into whole seconds.')
    b=int(round(b)); width=b*1_000_000_000
    if nominal>b:
        raise DataError('The plot bin must be at least as long as the normal sample interval.')
    if opts.aggregation=='activity' and opts.units not in {'binary','binary_flag'}:
        raise DataError('These raw tick rows contain samples. Choose 0/1 flags or activity counts. Seconds/minutes active require a file of measured interval totals.')
    if any(selected not in {NONE,AUTO,''} for selected in (opts.duration,opts.observed,opts.episodes)):
        raise DataError('Raw tick samples use elapsed time for their intervals. Leave duration, observed-time and episode columns unset.')
    values=activity_values(df[ac],opts,ac)
    if np.any(np.isinf(values) | (np.isfinite(values)&(values<0))):
        raise DataError(f'{ac}: activity values must be finite nonnegative numbers, or blank for missing measurements.')
    elapsed=numeric(df[elapsed_column],opts.decimal)
    kind='duration' if opts.units=='binary' and opts.aggregation=='activity' else 'numeric'
    unit='Activity seconds' if kind=='duration' else ('Active records' if opts.units=='binary_flag' else (opts.value_unit.strip() or 'Activity units'))
    aggregation='sum' if kind=='duration' or opts.units=='binary_flag' else opts.aggregation
    out=[]; total_bins=0
    for sid in pd.unique(ids):
        ix=ids.eq(sid).to_numpy(); gs=pd.unique(groups.loc[ix])
        if len(gs)!=1:
            raise DataError(f'{sid}: use one individual group per recording.')
        ns,clock=sample_clock(times.loc[ix],elapsed[ix],df.loc[ix,tc],str(sid))
        v=values[ix]; known=np.isfinite(v)
        gaps=np.diff(ns)
        nominal_ns=int(round(nominal*1e9))
        if nominal_ns<1:
            raise DataError('The normal sample interval is smaller than one nanosecond.')
        durations=np.full(len(ns),nominal_ns,dtype=np.int64)
        if len(gaps):
            durations[:-1]=np.where(gaps<=round(nominal_ns*1.5),gaps,nominal_ns)
        ends=ns+durations
        origin=int(ns[0]//width)*width
        count=int((int(ends[-1])-origin+width-1)//width)
        total_bins+=count
        if total_bins>MAX_MATRIX_CELLS:
            raise DataError('Raw sample import is too large. Increase the bin size or import a shorter recording.')
        index=(ns-origin)//width
        weights=np.bincount(index[known],minlength=count).astype(float)
        observed=sample_interval_totals(ns[known],ends[known],np.ones(int(known.sum())),origin,width,count)
        if kind=='duration':
            value=sample_interval_totals(ns[known],ends[known],v[known],origin,width,count)
            seconds=value.copy()
        else:
            value=np.bincount(index[known],weights=v[known],minlength=count).astype(float)
            if aggregation=='mean':
                value=np.divide(value,weights,out=np.full(count,np.nan),where=weights>0)
            seconds=np.full(count,np.nan)
        value[observed<=0]=np.nan
        seconds[observed<=0]=np.nan
        grid=pd.date_range(pd.Timestamp(origin),periods=count,freq=pd.Timedelta(seconds=b))
        normal=pd.DataFrame({'time':grid,'duration':float(b),'observed':observed,'seconds':seconds,
                             'episodes':np.nan,'value':value,'weight':weights})
        normal,dups=validate_frame(normal,str(sid),kind)
        source=dict(provenance)
        source['raw_samples']={'rows':int(len(v)),'measured_rows':int(known.sum()),
                               'timestamp_column':tc,'elapsed_column':elapsed_column,
                               'normal_interval_seconds':nominal,'gap_limit_seconds':nominal*1.5,
                               'source_bin_seconds':b,**clock}
        out.append(Subject(str(sid),str(gs[0]),normal,sources=[source],measure_kind=kind,
                           value_unit=unit,aggregation=aggregation))
        notes.append(f'{sid}: {len(v):,} raw samples retained in {count:,} bins of {b:g} seconds. ')
    notes.append('Raw sample coverage follows the elapsed clock. Gaps longer than 1.5 times the normal sample interval remain unknown after one normal interval; the last sample covers one normal interval. Missing measurements remain unknown.')
    if kind=='duration':
        notes.append('Active time is estimated by holding each selected state until the next sample within the coverage rule.')
    elif opts.units=='binary_flag':
        notes.append('Flags count active samples, not seconds')
    notes.append('The calendar origin is estimated assuming iso_time is truncated to its displayed precision')
    return out,notes


def decode_occupancy_logger(name: str, raw: bytes) -> tuple[list[Subject], list[str]]:
    
    df=read_table(raw)
    time_col=next((c for c in df if normal_name(c).startswith('date_time')),None)
    value_col=next((c for c in df if normal_name(c).startswith('occupancy')),None)
    if time_col is None or value_col is None:
        raise DataError('This is not a supported occupancy logger export.')
    times=pd.to_datetime(df[time_col],format='mixed',dayfirst=False,errors='coerce')
    values=numeric(df[value_col]); valid=np.isfinite(values)&(values>=0)&(values<=100)
    if times.isna().any() or not valid.all():
        raise DataError('Logger contains invalid timestamps or occupancy percentages outside 0–100.')
    if not times.is_monotonic_increasing:
        raise DataError('Logger timestamps are not chronological.')
    # Repeated minute timestamps do not identify the precise sample seconds.
    # Infer the modal samples/minute, only for complete interior minutes.
    counts=times.value_counts().sort_index()
    typical=int(counts.value_counts().idxmax())
    if typical<1 or 60%typical:
        raise DataError('Cannot infer a regular sample interval from repeated minute timestamps.')
    interval=60//typical
    # Map the first through Nth sample within each displayed minute to consecutive
    
    positions=df.groupby(times,sort=False).cumcount().to_numpy()
    if np.any(positions>=typical):
        raise DataError('Some minutes contain more samples than the inferred rate. The recording needs explicit sample timestamps.')
    starts=times+pd.to_timedelta(positions*interval,unit='s')
    base=starts.min().floor('min'); end=(starts.max()+pd.Timedelta(seconds=interval)).ceil('min')
    grid=pd.date_range(base,end,freq='min',inclusive='left')
    bucket=((starts-base).dt.total_seconds()//60).to_numpy(int)
    observed=np.bincount(bucket,weights=np.full(len(bucket),interval),minlength=len(grid)).astype(float)
    active=np.bincount(bucket,weights=values*interval/100,minlength=len(grid)).astype(float)
    active[observed==0]=np.nan
    normal=pd.DataFrame({'time':grid,'duration':60.,'observed':observed,'seconds':active,
                         'episodes':np.nan,'value':active,'weight':np.where(observed>0,1.,0.)})
    sid=snail_id_from_path(name)
    normal,_=validate_frame(normal,sid,'duration')
    source={'file':name,'sha256':hashlib.sha256(raw).hexdigest(),'clock':'wall',
            'logger_occupancy':{'inferred_sample_seconds':interval,'timestamp_precision':'minute',
                                'samples_per_complete_minute':typical}}
    notes=[f'{sid}: imported {len(df):,} occupancy readings; inferred {interval}-second sampling from {typical} records per minute.',
           'Occupancy percentages are converted to active seconds per nominal sample.']
    return [Subject(sid,'Ungrouped',normal,sources=[source])],notes


def decode_source(name: str, raw: bytes, opts: ImportOptions) -> tuple[list[Subject],list[str]]:
    df = read_table(raw)
    if any(normal_name(c).startswith('occupancy') for c in df.columns) and any(normal_name(c).startswith('date_time') for c in df.columns):
        return decode_occupancy_logger(name,raw)
    if opts.data_mode not in {'binned','samples','events'} or opts.aggregation not in {'activity','sum','mean'}:
        raise DataError('Choose a supported data type and calculation.')
    if opts.timestamp_kind not in {'calendar','unix_s','unix_ms','elapsed_s','elapsed_min'}:
        raise DataError('Unknown timestamp format.')
    if opts.units not in {'seconds','minutes','fraction','percent','binary','binary_flag'}:
        raise DataError('Unknown activity units.')
    times, notes = source_times(df,opts)
    sc = resolve_column(df,opts.subject,('snail_id','subject_id','animal_id','subject','snail','animal','id'))
    gc = resolve_column(df,opts.group,('group','population','treatment'))
    stem = snail_id_from_path(name)
    ids = df[sc].fillna('').astype(str).str.strip() if sc else pd.Series([stem]*len(df))
    groups = df[gc].fillna('Ungrouped').astype(str).str.strip().replace('','Ungrouped') if gc else pd.Series(['Ungrouped']*len(df))
    if ids.eq('').any():
        raise DataError(f'{name}: empty individual IDs.')
    if not sc:
        notes.append(f'{name}: ID {stem} extracted from the filename; no ID column required.')
    provenance = {'file':name,'sha256':hashlib.sha256(raw).hexdigest(),
                  'clock':'utc' if opts.timestamp_kind.startswith('unix') else opts.time_mode,
                  'import_options':asdict(opts)}
    out = []
    if opts.data_mode == 'events':
        if not opts.continuous_events or not opts.recording_start or not opts.recording_end:
            raise DataError('Timestamp-only events need declared recording start/end AND confirmation of continuous logging. ')
        start,end = parse_datetime(opts.recording_start),parse_datetime(opts.recording_end)
        if end<=start:
            raise DataError('Event recording end must be after its start.')
        b = float(opts.raw_bin_minutes)*60
        if not math.isfinite(b) or b<1 or abs(b-round(b))>1e-7 or DAY_SECONDS%int(round(b)):
            raise DataError('Event bin size must divide 24 hours into whole seconds.')
        b=int(round(b)); width=b*1_000_000_000
        origin = pd.Timestamp((start.value//width)*width)
        count = int(math.ceil((end-origin).total_seconds()/b))
        if count * ids.nunique() > MAX_MATRIX_CELLS:
            raise DataError('Event import is too large. Increase its bin size or shorten the declared recording.')
        grid = pd.date_range(origin,periods=count,freq=pd.Timedelta(seconds=b))
        gns=grid.to_numpy(dtype='datetime64[ns]').astype(np.int64)
        obs=(np.minimum(gns+width,end.value)-np.maximum(gns,start.value))/1e9
        if ((times<start)|(times>=end)).any():
            raise DataError('Some event timestamps are outside the declared recording [start, end). Check the bounds.')
        for sid in pd.unique(ids):
            ix=ids.eq(sid).to_numpy()
            gs=pd.unique(groups.loc[ix])
            if len(gs)!=1: raise DataError(f'{sid}: use one individual group per recording.')
            ns=times.loc[ix].to_numpy(dtype='datetime64[ns]').astype(np.int64)
            values=np.bincount((ns-origin.value)//width,minlength=count).astype(float)
            normal=pd.DataFrame({'time':grid,'duration':float(b),'observed':obs,
                                 'seconds':np.nan,'episodes':np.nan,'value':values,'weight':1.})
            normal,dups=validate_frame(normal,str(sid),'events')
            out.append(Subject(str(sid),str(gs[0]),normal,sources=[provenance],
                               measure_kind='events',value_unit='Events',aggregation='sum'))
        notes.append('Event counts are not activity duration. Identical event timestamps count as separate events. Zero-event bins are known ONLY because continuous recording was explicitly declared. Import separate recording sessions separately.')
        notes.append(f'Events accumulated into {b:g}-second source bins. Reimport the raw events to use finer bins later.')
        return out,notes
    ac=resolve_column(df,opts.activity,('motion_seconds','activity_seconds','active_seconds','motion_minutes','activity_minutes','activity','active_fraction','activity_percent','value','motion_flag','motion','state'),True)
    raw_clock=resolve_column(df,opts.timestamp,TIMESTAMP_COLUMNS,False)
    raw_elapsed=resolve_column(df,AUTO,('t_since_start_s',),False)
    if (opts.data_mode=='samples' and opts.timestamp_kind=='calendar' and opts.date_column in {NONE,AUTO,''}
            and raw_clock and normal_name(raw_clock)=='iso_time' and raw_elapsed):
        return decode_tick_samples(df,times,ids,groups,opts,provenance,ac,raw_clock,raw_elapsed,notes)
    dc=resolve_column(df,opts.duration,('bin_seconds','bin_duration_seconds','bin_duration_sec','sample_seconds'))
    oc=resolve_column(df,opts.observed,('observed_seconds','recorded_seconds','coverage_seconds'))
    ec=resolve_column(df,opts.episodes,('episodes_started','episode_starts','episodes'))
    if opts.data_mode=='samples':
        d=float(opts.sample_seconds)
        if not math.isfinite(d) or d<=0: raise DataError('Declare a positive sample duration in seconds.')
        durations=numeric(df[dc],opts.decimal) if dc else np.full(len(df),d)
        notes.append(f'Raw samples: each value represents its declared sample interval ({d:g} s fallback), NOT the gap to the next row.')
    elif dc:
        durations=numeric(df[dc],opts.decimal)
    else:
        bs=resolve_column(df,AUTO,('bin_start_sec',)) if opts.duration==AUTO else None
        be=resolve_column(df,AUTO,('bin_end_sec',)) if opts.duration==AUTO else None
        if bs and be:
            durations=numeric(df[be],opts.decimal)-numeric(df[bs],opts.decimal)
        else:
            d=float(opts.fallback_seconds)
            if not math.isfinite(d) or d<=0: raise DataError('Declare a positive source bin duration.')
            durations=np.full(len(df),d)
            notes.append(f'{name}: no source duration column; using declared {d:g} s, not gaps between timestamps.')
    observed=numeric(df[oc],opts.decimal) if oc else durations.copy()
    observed=np.where(np.isfinite(observed),observed,0.)
    value=activity_values(df[ac],opts,ac)
    episodes=numeric(df[ec],opts.decimal) if ec else np.full(len(df),np.nan)
    kind='duration' if opts.aggregation=='activity' and opts.units!='binary_flag' else 'numeric'
    if kind=='duration':
        hints={'motion_minutes':'minutes','activity_minutes':'minutes','active_fraction':'fraction','activity_percent':'percent'}
        expected=hints.get(normal_name(ac))
        if expected and opts.units!=expected:
            raise DataError(f'{ac} suggests {expected}. Set the activity units accordingly.')
        if opts.units=='seconds': seconds=value.copy()
        elif opts.units=='minutes': seconds=value*60
        else:
            top=100 if opts.units=='percent' else 1
            if np.any(np.isfinite(value)&((value<0)|(value>top))):
                raise DataError(f'{ac}: selected units require values from 0 to {top}.')
            if opts.units=='binary' and np.any(np.isfinite(value)&~np.isin(value,[0,1])):
                raise DataError('Binary activity samples must be 0 or 1. Choose fraction for intermediate values.')
            seconds=value/top*observed
        value=seconds.copy()
        unit='Activity seconds'; aggregation='sum'
        if opts.units=='binary':
            notes.append('Each active state is treated as lasting for its declared observed interval. Use occurrence flags if 1 only means activity happened at some point in the interval.')
    else:
        seconds=np.full(len(df),np.nan)
        unit='Active records' if opts.units=='binary_flag' else (opts.value_unit.strip() or 'Activity units')
        aggregation='sum' if opts.units=='binary_flag' else opts.aggregation
        if opts.units=='binary_flag':
            notes.append('Flags count active records. A flag does not specify activity duration and is not converted to seconds.')
        if aggregation=='mean' and opts.data_mode=='binned':
            notes.append('Binned numeric means are combined with equal source-bin weights. Import raw samples when unequal sample counts require sample-weighted means.')
    normal=pd.DataFrame({'time':times,'duration':durations,'observed':observed,
                         'seconds':seconds,'episodes':episodes,'value':value,
                         'weight':np.where(np.isfinite(value),1.,0.)})
    elapsed=resolve_column(df,AUTO,('bin_start_sec','elapsed_seconds'))
    if opts.timestamp_kind in {'elapsed_s','elapsed_min'}:
        normal['elapsed']=(times-parse_datetime(opts.recording_start)).dt.total_seconds().to_numpy(float)
    elif elapsed:
        normal['elapsed']=numeric(df[elapsed],opts.decimal)
    for sid in pd.unique(ids):
        ix=ids.eq(sid).to_numpy(); gs=pd.unique(groups.loc[ix])
        if len(gs)!=1: raise DataError(f'{sid}: multiple groups within an individual. Group is a subject assignment, not a time-varying treatment column.')
        data,dups=validate_frame(normal.loc[ix],str(sid),kind)
        if dups: notes.append(f'{sid}: removed {dups} identical duplicate interval rows.')
        out.append(Subject(str(sid),str(gs[0]),data,sources=[provenance],measure_kind=kind,value_unit=unit,aggregation=aggregation))
        notes.append(f'{sid}: {len(data):,} intervals; typical interval {float(data.duration.mode().iloc[0]):g} s; {datetime_text(data.time.min())} to {datetime_text((data.time+pd.to_timedelta(data.duration,unit="s")).max())}.')
    if not oc: notes.append('Present numeric intervals are treated as fully observed. Missing rows or invalid activity values remain missing, not zero activity.')
    return out,notes


def merge_subjects(existing: list[Subject],incoming: list[Subject]) -> tuple[list[Subject],list[str]]:
    merged={s.id:replace(s,data=s.data.copy(),sources=list(s.sources)) for s in existing}
    notes=[]
    for s in incoming:
        if s.id not in merged:
            merged[s.id]=replace(s,data=s.data.copy(),sources=list(s.sources)); continue
        old=merged[s.id]
        if (old.measure_kind,old.value_unit,old.aggregation)!=(s.measure_kind,s.value_unit,s.aggregation):
            raise DataError(f'{s.id}: cannot merge different measurement types/units. Use distinct IDs or import consistent data.')
        if len({v.get('clock','wall') for v in old.sources+s.sources})>1:
            raise DataError(f'{s.id}: different clock bases cannot be merged.')
        if old.group!=s.group and old.group!='Ungrouped' and s.group!='Ungrouped':
            raise DataError(f'{s.id}: conflicting groups: {old.group} / {s.group}.')
        old.data,dups=validate_frame(pd.concat([old.data,s.data],ignore_index=True),s.id,s.measure_kind)
        if old.group=='Ungrouped': old.group=s.group
        old.sources=list({(x['file'],x['sha256']):x for x in old.sources+s.sources}.values())
        if dups: notes.append(f'{s.id}: ignored {dups} identical source intervals from overlapping/repeated imports.')
    return list(merged.values()),notes


def metric_label(cfg: Settings,subject: Subject) -> str:
    if cfg.metric=='auto':
        if subject.measure_kind=='duration': return 'Activity seconds / bin'
        return subject.value_unit+(' · mean per source row' if subject.aggregation=='mean' else ' / bin')
    return METRICS[cfg.metric]


def threshold_units(subject: Subject | None = None) -> str:
    if subject is None: return 'imported activity units'
    if subject.measure_kind=='duration': return 'seconds / bin'
    return subject.value_unit+(' / source row' if subject.aggregation=='mean' else ' / bin')


def filtered_subject(subject: Subject,cfg: Settings) -> tuple[Subject,pd.Timestamp,int]:
    frame=subject.data.copy()
    if cfg.timezone_offset_min:
        frame['time'] += pd.Timedelta(minutes=cfg.timezone_offset_min)
    first=frame.time.min()
    end=(frame.time+pd.to_timedelta(frame.duration,unit='s')).max()
    lo=first+pd.Timedelta(hours=cfg.trim_first_hours)
    hi=end-pd.Timedelta(hours=cfg.trim_last_hours)
    if cfg.window_start: lo=max(lo,parse_datetime(cfg.window_start))
    if cfg.window_end: hi=min(hi,parse_datetime(cfg.window_end))
    starts=frame.time
    ends=starts+pd.to_timedelta(frame.duration,unit='s')
    keep=(starts>=lo)&(starts<hi)
    if cfg.crop_policy=='complete': keep &= ends<=hi
    for day in parse_excluded_dates(cfg.excluded_dates):
        if cfg.crop_policy=='complete':
            keep &= ~((starts<day+pd.Timedelta(days=1))&(ends>day))
        else:
            keep &= starts.dt.normalize()!=day
    return replace(subject,data=frame.loc[keep].copy()),first,int((~keep).sum())


def bin_subject(subject: Subject,origin: pd.Timestamp,days: int,cfg: Settings) -> dict[str,np.ndarray]:
    b=cfg.bin_seconds; width=b*1_000_000_000
    count=days*(DAY_SECONDS//b)
    df=subject.data
    starts=df.time.to_numpy(dtype='datetime64[ns]').astype(np.int64)
    rel=starts-origin.value
    keep=(rel>=0)&(rel<count*width)
    rel=rel[keep]; df=df.loc[keep]
    idx=(rel//width).astype(np.int64)
    durations=np.rint(df.duration.to_numpy(float)*1e9).astype(np.int64)
    if len(df) and np.any(durations>width+1000):
        raise DataError(f'{subject.name}: display bins are smaller than a source interval ({df.duration.max():g} s). Choose a compatible larger bin; accumulated totals cannot supply finer timing.')
    if cfg.bin_policy=='strict' and len(df) and np.any((rel+durations-1)//width!=idx):
        raise DataError(f'{subject.name}: an interval crosses a display-bin boundary. Use whole rows by start, or align days to recording start.')
    val,secs,obs,ep,weights=(df[c].to_numpy(float) for c in ('value','seconds','observed','episodes','weight'))
    known=np.isfinite(val)&(obs>0)
    def accum(w): return np.bincount(idx,weights=w,minlength=count).astype(float)
    observed=accum(np.where(known,obs,0))
    if np.any(observed>b+1e-5):
        raise DataError(f'{subject.name}: this grid collects more observed time than a bin can contain. Use a multiple of the source interval or recording-start alignment. Values were not clipped.')
    total=accum(np.where(known&np.isfinite(secs),secs,0))
    total[(observed<=0)|(subject.measure_kind!='duration')]=np.nan
    episode_obs=accum(np.where(known&np.isfinite(ep),obs,0))
    episodes=accum(np.where(known&np.isfinite(ep),ep,0))
    episodes[(observed<=0)|~np.isclose(episode_obs,observed,rtol=0,atol=1e-6)]=np.nan
    if subject.aggregation=='mean':
        w=accum(np.where(known,weights,0))
        numerator=accum(np.where(known,val*weights,0))
        values=np.divide(numerator,w,out=np.full(count,np.nan),where=w>0)
    else:
        values=accum(np.where(known,val,0)); values[observed<=0]=np.nan
    valid=(observed>0)&(observed+1e-6>=b*cfg.minimum_coverage/100)
    classified=np.where(valid,values,np.nan)
    binary=threshold_activity(classified,cfg.threshold,cfg.comparison)
    if cfg.metric=='seconds': values=total.copy()
    elif cfg.metric=='minutes': values=total/60
    elif cfg.metric=='percent': values=np.divide(total*100,observed,out=np.full(count,np.nan),where=observed>0)
    elif cfg.metric=='episodes': values=episodes.copy()
    raw_values=values.copy()
    if cfg.metric=='active_percent':
        raw_values=threshold_activity(raw_values,0,'ge')*100
        values=binary*100
    else:
        values=np.where(valid,values,np.nan)
        values=np.where(np.isfinite(values)&(binary==0),0.,values)
    shape=(days,DAY_SECONDS//b)
    return {k:v.reshape(shape) for k,v in {'seconds':total,'observed':observed,'episodes':episodes,
                                          'values':values,'binary':binary,'raw_values':raw_values}.items()}


def build_analysis(subjects: list[Subject],cfg: Settings,group_colours: dict[str,str]) -> Analysis:
    cfg=replace(cfg,treatments=[dict(t) for t in cfg.treatments]); cfg.validate()
    chosen=[s for s in subjects if s.included]
    if not chosen: raise DataError('Import data and include at least one individual.')
    signatures={(s.measure_kind,s.value_unit,s.aggregation) for s in chosen}
    if len(signatures)>1:
        raise DataError('The included individuals use different measurement types or units..')
    clocks={p.get('clock','wall') for s in chosen for p in s.sources}
    if len(clocks)>1: raise DataError('Included recordings have different clock bases. Reimport them consistently.')
    if chosen[0].measure_kind!='duration' and cfg.metric in {'seconds','minutes','percent','episodes'}:
        raise DataError('These files contain values/event counts, not time active. Choose Automatic imported units or the active-individual percentage.')
    filtered=[]; originals={}; notes=[]
    for s in chosen:
        f,anchor,dropped=filtered_subject(s,cfg)
        filtered.append(f); originals[s.id]=anchor
        if dropped: notes.append(f'{s.id}: {dropped:,} complete source intervals omitted by date/time filters. No activity was fractionally split.')
    nonempty=[s for s in filtered if not s.data.empty]
    if not nonempty: raise DataError('No intervals remain in the plotting range. Check the start/end, trimming and excluded dates.')
    offset=pd.Timedelta(seconds=cfg.start_seconds)
    if cfg.alignment=='clock24':
        first=min((s.data.time.min()-offset).normalize() for s in nonempty)
        last=max((s.data.time.max()-offset).normalize() for s in nonempty)
        days=int((last-first).days)+1
        origin_map={s.id:first+offset for s in filtered}
        original_first=min((t-offset).normalize() for t in originals.values())
        base_day=int((first-original_first).days)
        dates=pd.date_range(first,periods=days+1,freq='D')
    else:
        lo=min(int((s.data.time.min()-originals[s.id]).total_seconds()//DAY_SECONDS) for s in nonempty)
        hi=max(int((s.data.time.max()-originals[s.id]).total_seconds()//DAY_SECONDS) for s in nonempty)
        days=hi-lo+1; base_day=lo
        origin_map={s.id:originals[s.id]+pd.Timedelta(days=lo) for s in filtered}
        dates=pd.date_range('1970-01-01',periods=days+1,freq='D')
    ngroups=len({s.group for s in filtered})
    if (len(filtered)+ngroups)*(days+1)*(DAY_SECONDS//cfg.bin_seconds)>MAX_MATRIX_CELLS:
        raise DataError('This view would use too much memory. Shorten the range, increase bin size or include fewer individuals.')
    panels={}; by_group={}
    for s in filtered:
        a=bin_subject(s,origin_map[s.id],days+1,cfg)
        key=f'subject:{s.id}'
        p=Panel(key,s.name,s.group,False,s.colour or group_colours.get(s.group,NAMED_COLOURS['Ink']),
                a['seconds'],a['observed'],a['episodes'],a['values'],a['binary'],a['binary']*100,
                np.isfinite(a['values']).astype(int),np.isfinite(a['binary']).astype(int),1,
                origin_map[s.id],s.measure_kind,metric_label(cfg,s),a['raw_values'],threshold_units(s))
        panels[key]=p; by_group.setdefault(s.group,[]).append(p)
    for group,members in by_group.items():
        active,n_binary=finite_mean(np.stack([p.binary for p in members]))
        active*=100
        g_binary=threshold_activity(active,cfg.group_active_percent,'ge')
        g_binary[n_binary<cfg.minimum_subjects]=np.nan
        active[n_binary<cfg.minimum_subjects]=np.nan
        values,n=finite_mean(np.stack([p.values for p in members]))
        values[n<cfg.minimum_subjects]=np.nan
        unfiltered,_=finite_mean(np.stack([np.where(np.isfinite(p.values),p.unfiltered_values,np.nan) for p in members]))
        unfiltered[n<cfg.minimum_subjects]=np.nan
        secs,_=finite_mean(np.stack([np.where(np.isfinite(p.binary),p.seconds,np.nan) for p in members]))
        obs,_=finite_mean(np.stack([np.where(np.isfinite(p.binary),p.observed,np.nan) for p in members]))
        ep,_=finite_mean(np.stack([np.where(np.isfinite(p.binary),p.episodes,np.nan) for p in members]))
        for a in (secs,obs,ep): a[n_binary<cfg.minimum_subjects]=np.nan
        origins={p.origin for p in members}; origin=next(iter(origins)) if len(origins)==1 else None
        key=f'group:{group}'
        panels[key]=Panel(key,group,group,True,group_colours.get(group,NAMED_COLOURS['Ink']),secs,obs,ep,values,g_binary,active,n,n_binary,len(members),origin,members[0].measure_kind,members[0].unit,unfiltered,members[0].threshold_units)
    rows=np.arange(days,dtype=int)
    if not any(np.isfinite(p.binary[rows]).any() for p in panels.values() if not p.is_group):
        raise DataError('No usable bins remain for the selected activity measure and observation coverage. Check the activity measure, minimum coverage and bin size.')
    maxima={False:1.,True:1.}
    for is_group in maxima:
        vals=[displayed(p.values,days,cfg.double_plot,rows) for p in panels.values() if p.is_group==is_group]
        maxima[is_group]=max([float(np.nanmax(x)) for x in vals if np.isfinite(x).any()]+[1e-6])
    if cfg.metric in {'percent','active_percent'}: maxima={False:100.,True:100.}
    notes.extend([
        ' ',
        'Start is inclusive, end exclusive. ',
        'Groups are equal-weight per-bin means of available individuals.',
        '',
        '',
        '',
    ])
    if cfg.schedule_enabled and cfg.schedule_mode=='multiple':
        notes.append('Gaps between treatment periods are Unassigned, not assumed darkness. Treatment ends are exclusive.')
    return Analysis(cfg,dates,days,panels,maxima,notes,rows,base_day)

def phase_name(t: Treatment, phase: str) -> str:
    if phase=='Light':
        name,lux=t.light_name or 'Light',t.light_lux
    elif phase=='Dark':
        name,lux=t.dark_name or 'Dark',t.dark_lux
    else:
        return 'No annotation'
    return name+(f' · {float(lux):g} lux' if lux is not None else '')


def schedule_segments(start: pd.Timestamp,end: pd.Timestamp,cfg: Settings):
    """Exact half-open phase spans; gaps in the protocol are deliberately unassigned."""
    if not cfg.schedule_enabled: return []
    segments=[]
    for t in validated_treatments(cfg):
        lo=max(start,parse_datetime(t.start)) if t.start else start
        hi=min(end,parse_datetime(t.end)) if t.end else end
        if hi<=lo or t.cycle=='NONE': continue
        edges={lo,hi}
        if t.cycle=='LD':
            on,off=parse_clock(t.lights_on),parse_clock(t.lights_off)
            for date in pd.date_range(lo.normalize(),hi.normalize(),freq='D'):
                for sec in (on,off):
                    edge=date+pd.Timedelta(seconds=sec)
                    if lo<edge<hi: edges.add(edge)
        edges=sorted(edges)
        for a,b in zip(edges[:-1],edges[1:]):
            if t.cycle=='DD': light=False
            elif t.cycle=='LL': light=True
            else:
                sec=a.hour*3600+a.minute*60+a.second+a.microsecond/1e6
                light=(on<=sec<off) if on<off else (sec>=on or sec<off)
            phase='Light' if light else 'Dark'
            segments.append((a,b,t,phase,t.light_colour if light else t.dark_colour))
    return sorted(segments,key=lambda x:x[0])


def classify_schedule(times: pd.DatetimeIndex,durations: np.ndarray,cfg: Settings) -> pd.DataFrame:
    """Label display slots, keeping boundary-crossing slots in a separate category."""
    n=len(times)
    period=np.full(n,'Unassigned',dtype=object)
    phase=np.full(n,'Unassigned',dtype=object)
    illumination=np.full(n,'',dtype=object); lux=np.full(n,np.nan)
    if not cfg.schedule_enabled:
        period[:]='Not specified'; phase[:]='Not specified'
        return pd.DataFrame({'treatment':period,'phase':phase,'illumination':illumination,'lux':lux})
    ns=times.to_numpy(dtype='datetime64[ns]').astype(np.int64)
    duration_ns=np.rint(np.asarray(durations)*1e9).astype(np.int64)
    end=ns+duration_ns
    secs=times.hour.to_numpy()*3600+times.minute.to_numpy()*60+times.second.to_numpy()+times.microsecond.to_numpy()/1e6
    boundary=np.zeros(n,dtype=bool)
    for t in validated_treatments(cfg):
        lo=parse_datetime(t.start).value if t.start else np.iinfo(np.int64).min
        hi=parse_datetime(t.end).value if t.end else np.iinfo(np.int64).max
        inside=(ns>=lo)&(ns<hi)
        period[inside]=t.name
        if t.start: boundary |= (ns<lo)&(end>lo)
        if t.end: boundary |= (ns<hi)&(end>hi)
        if t.cycle=='NONE':
            phase[inside]='No annotation'; continue
        if t.cycle=='LD':
            on,off=parse_clock(t.lights_on),parse_clock(t.lights_off)
            lit=((secs>=on)&(secs<off)) if on<off else ((secs>=on)|(secs<off))
            d1=(on-secs)%DAY_SECONDS; d2=(off-secs)%DAY_SECONDS
            d1=np.where(d1<1e-9,DAY_SECONDS,d1); d2=np.where(d2<1e-9,DAY_SECONDS,d2)
            boundary |= inside & (np.asarray(durations)>np.minimum(d1,d2)+1e-6)
        else:
            lit=np.full(n,t.cycle=='LL')
        for light,ph in ((True,'Light'),(False,'Dark')):
            mask=inside&(lit==light)
            phase[mask]=ph
            illumination[mask]=t.light_name if light else t.dark_name
            level=t.light_lux if light else t.dark_lux
            lux[mask]=np.nan if level is None else float(level)
    period[boundary]='Boundary interval'; phase[boundary]='Boundary interval'
    illumination[boundary]=''; lux[boundary]=np.nan
    return pd.DataFrame({'treatment':period,'phase':phase,'illumination':illumination,'lux':lux})


def get_cmap(cfg: Settings,colour: str):
    if cfg.cmap=='group': cm=LinearSegmentedColormap.from_list('individual_tint',[cfg.background_colour,colour])
    elif cfg.cmap=='custom': cm=LinearSegmentedColormap.from_list('custom_tint',[cfg.low_colour,cfg.high_colour])
    else: cm=colormaps[cfg.cmap].copy()
    if cfg.reverse_cmap: cm=cm.reversed()
    cm.set_bad(cfg.missing_colour)
    return cm


def panel_vmax(analysis: Analysis,panel: Panel) -> float:
    cfg=analysis.settings
    if cfg.scale_mode=='manual': return float(cfg.vmax)
    if cfg.scale_mode=='shared': return analysis.shared_vmax[panel.is_group]
    a=displayed(panel.values,analysis.shown_days,cfg.double_plot,analysis.row_indices)
    return max(float(np.nanmax(a)),1e-6) if np.isfinite(a).any() else 1.


def draw_schedule(ax,analysis: Analysis,p: Panel,heatmap: bool) -> list[Patch]:
    cfg=analysis.settings
    if not cfg.schedule_enabled or p.origin is None or cfg.shade_mode=='none' or cfg.shade_opacity<=0:
        return []
    if heatmap and cfg.heatmap_schedule=='none': return []
    down=cfg.orientation=='days_down'; keys={}
    for y,row in enumerate(analysis.row_indices):
        start=p.origin+pd.Timedelta(days=int(row))
        end=start+pd.Timedelta(days=2 if cfg.double_plot else 1)
        for a,b,t,phase,colour in schedule_segments(start,end,cfg):
            if cfg.shade_mode not in {'both',phase.lower()}: continue
            left=(a-start).total_seconds()/3600; right=(b-start).total_seconds()/3600
            rect=Rectangle((left,y),right-left,1) if down else Rectangle((y,left),1,right-left)
            rect.set(facecolor=colour,edgecolor='none',linewidth=0,alpha=cfg.shade_opacity,
                     antialiased=False,zorder=2)
            ax.add_patch(rect)
            label=f'{t.name}: {phase_name(t,phase)}' if cfg.schedule_mode=='multiple' else phase
            keys[(label,colour)]=Patch(facecolor=colour,edgecolor='none',alpha=cfg.shade_opacity,label=label)
    return list(keys.values())


SOLAR_LEVELS = (
    ('solar_horizon', 0.0, 'Horizon (0°)', '-'),
    ('solar_civil', -6.0, 'Civil twilight (−6°)', '--'),
    ('solar_nautical', -12.0, 'Nautical twilight (−12°)', '-.'),
    ('solar_astronomical', -18.0, 'Astronomical twilight (−18°)', ':'),
)

@lru_cache(maxsize=16)
def solar_observer(latitude, longitude):
    try:
        from astral import Observer
    except ImportError as exc:
        raise DataError('Solar curves require Astral. Install with: pip install astral') from exc
    return Observer(latitude=latitude,longitude=longitude)

@lru_cache(maxsize=20000)
def solar_crossings(day_iso, latitude, longitude, tz_name, basis):
    from astral.sun import elevation
    observer=solar_observer(latitude,longitude)
    zone=ZoneInfo(tz_name)
    if basis=='utc': plot_zone=timezone.utc
    elif basis=='standard':
        january=datetime(int(day_iso[:4]),1,15,12,tzinfo=zone)
        plot_zone=timezone(january.utcoffset()-january.dst())
    else: plot_zone=zone
    day=datetime.fromisoformat(day_iso)
    start=day.replace(tzinfo=plot_zone)
    minutes=np.arange(0,1441,5,dtype=int)
    elevations=np.array([elevation(observer,start+timedelta(minutes=int(m))) for m in minutes],dtype=float)
    result={}
    for key,level,label,linestyle in SOLAR_LEVELS:
        shifted=elevations-level
        crossings=[]
        for i in range(len(minutes)-1):
            a,b=shifted[i],shifted[i+1]
            if not (np.isfinite(a) and np.isfinite(b)): continue
            if (a<0<=b) or (a>0>=b):
                fraction=-a/(b-a) if b!=a else 0.
                hour=float((minutes[i]+fraction*(minutes[i+1]-minutes[i]))/60)
                crossings.append((hour,'rise' if b>a else 'set'))
        result[key]=crossings
    return result

def draw_solar(ax,analysis,p):
    cfg=analysis.settings
    if not cfg.solar_enabled or p.origin is None or cfg.alignment!='clock24': return []
    down=cfg.orientation=='days_down'
    chosen=[(k,lab,ls,getattr(cfg,k+'_colour')) for k,level,lab,ls in SOLAR_LEVELS if getattr(cfg,k)]
    if not chosen: return []
    cache={}
    def get(day):
        key=day.date().isoformat()
        if key not in cache:
            cache[key]=solar_crossings(key,round(float(cfg.solar_latitude),6),round(float(cfg.solar_longitude),6),cfg.solar_timezone,cfg.solar_time_basis)
        return cache[key]
    for key,label,ls,colour in chosen:
        for half in range(2 if cfg.double_plot else 1):
            for direction in ('rise','set'):
                xs=[]; ys=[]
                for i,row in enumerate(analysis.row_indices):
                    row_start=p.origin+pd.Timedelta(days=int(row)+half)
                    candidates=[]
                    for day in pd.date_range(row_start.normalize()-pd.Timedelta(days=1),row_start.normalize()+pd.Timedelta(days=1),freq='D'):
                        for crossing,kind in get(day)[key]:
                            if kind!=direction: continue
                            moment=day+pd.Timedelta(hours=crossing)
                            hour=(moment-row_start).total_seconds()/3600
                            if 0<=hour<24: candidates.append(hour)
                    xs.append((candidates[0]+24*half) if candidates else np.nan)
                    ys.append(i+.5)
                x=np.asarray(xs,dtype=float)
                jumps=np.where(np.isfinite(x[1:]) & np.isfinite(x[:-1]) & (np.abs(np.diff(x))>3))[0]
                for j in jumps: x[j+1]=np.nan
                ax.plot(x if down else ys,ys if down else x,color=colour,linestyle=ls,linewidth=1.35,zorder=8,
                        label=label if half==0 and direction=='rise' else '_nolegend_')
    from matplotlib.lines import Line2D
    return [Line2D([0],[0],color=c,linestyle=ls,linewidth=1.5,label=label) for k,label,ls,c in chosen]

def layout_actogram(fig: Figure,ax,colourbar,handles,cfg: Settings,days: int) -> None:
    import textwrap
    down=cfg.orientation=='days_down'
    canvas=FigureCanvasAgg(fig)
    canvas.draw()
    renderer=canvas.get_renderer()
    dpi=fig.dpi
    box=ax.get_window_extent(renderer)
    tight=ax.get_tightbbox(renderer)
    left=max(0.,(box.x0-tight.x0)/dpi)
    right=max(0.,(tight.x1-box.x1)/dpi)
    bottom=max(0.,(box.y0-tight.y0)/dpi)
    top=max(0.,(tight.y1-box.y1)/dpi)
    day_extent=days*float(cfg.row_height)
    plot_height=day_extent if down else 6.5
    bar_gap=.20
    bar_width=.14
    if colourbar is not None:
        bar_box=colourbar.ax.get_window_extent(renderer)
        bar_tight=colourbar.ax.get_tightbbox(renderer)
        right=max(right,bar_gap+bar_width+max(0.,(bar_tight.x1-bar_box.x1)/dpi))
        label_height=colourbar.ax.yaxis.label.get_window_extent(renderer).height/dpi
        overflow=max(0.,(label_height-plot_height)/2)
        top=max(top,overflow)
        bottom=max(bottom,overflow)
    pad=.16
    width=float(cfg.figure_width) if down else day_extent+left+right+2*pad
    legend=None
    legend_height=0.
    legend_gap=0.
    if handles and cfg.show_schedule_legend:
        font=max(6,cfg.font_size-2)
        labels=[h.get_label() for h in handles]
        columns=2 if width>=5.5 or width>=2.3 and max(map(len,labels))<18 else 1
        characters=max(10,int(((width-2*pad)/columns-.55)/(font/72*.52)))
        labels=[textwrap.fill(label,width=characters,break_long_words=True,break_on_hyphens=False) for label in labels]
        legend=fig.legend(handles=handles,labels=labels,loc='lower center',bbox_to_anchor=(.5,.01),
                          ncol=columns,frameon=False,fontsize=font,labelcolor=cfg.ink_colour)
        canvas.draw()
        legend_box=legend.get_window_extent(canvas.get_renderer())
        legend_height=legend_box.height/dpi
        width=max(width,legend_box.width/dpi+2*pad)
        legend_gap=.10
    plot_width=max(.1,width-left-right-2*pad) if down else day_extent
    width=max(width,plot_width+left+right+2*pad)
    height=plot_height+top+bottom+2*pad+legend_height+legend_gap
    x=pad+left+(width-plot_width-left-right-2*pad)/2
    y=pad+legend_height+legend_gap+bottom
    fig.set_size_inches(width,height)
    ax.set_position([x/width,y/height,plot_width/width,plot_height/height])
    if colourbar is not None:
        colourbar.ax.set_axes_locator(None)
        colourbar.ax.set_box_aspect(None)
        colourbar.ax.set_aspect('auto')
        colourbar.ax.set_position([(x+plot_width+bar_gap)/width,y/height,bar_width/width,plot_height/height])
        if down and plot_height<1.6:
            from matplotlib.ticker import MaxNLocator
            colourbar.locator=MaxNLocator(nbins=max(2,int(plot_height*3)),min_n_ticks=2)
            colourbar.update_ticks()
    if legend is not None:
        legend.set_bbox_to_anchor((.5,pad/height))



def make_figure(analysis: Analysis,key: str) -> Figure:
    cfg,p=analysis.settings,analysis.panels[key]
    days=analysis.shown_days; hours=48 if cfg.double_plot else 24
    down=cfg.orientation=='days_down'
    height=min(32,max(3.,1.6+days*cfg.row_height)) if down else 8.5
    width=cfg.figure_width if down else min(30,max(4.,2.+days*cfg.row_height))
    fig=Figure(figsize=(width,height),facecolor=cfg.background_colour)
    ax=fig.add_subplot(111); ax.set_facecolor(cfg.background_colour)
    vals=displayed(p.values,days,cfg.double_plot,analysis.row_indices)
    binary=displayed(p.binary,days,cfg.double_plot,analysis.row_indices)
    extent=(0,hours,days,0) if down else (0,days,hours,0)
    vmax=panel_vmax(analysis,p); cm=get_cmap(cfg,p.colour); norm=Normalize(0,vmax,clip=True)
    coloured=cfg.style=='heatmap' or cfg.mark_colour_mode=='intensity'
    basecolour=p.colour if cfg.use_group_colours else cfg.ink_colour
    colourbar=None
    if cfg.style=='heatmap':
        image=ax.imshow(np.ma.masked_invalid(vals if down else vals.T),origin='upper',extent=extent,
                        aspect='auto',interpolation='nearest',cmap=cm,vmin=0,vmax=vmax,rasterized=True)
    else:
        missing=~np.isfinite(binary if cfg.style in {'dots','ticks','hticks'} else vals)
        maskcm=ListedColormap([cfg.missing_colour]); maskcm.set_bad((0,0,0,0))
        image=ax.imshow(np.ma.masked_invalid(np.where(missing if down else missing.T,1.,np.nan)),
                        origin='upper',extent=extent,aspect='auto',interpolation='nearest',
                        cmap=maskcm,vmin=0,vmax=1,zorder=1,rasterized=True)
        isclassic=cfg.style in {'dots','ticks','hticks'}
        valid=(binary==1) if isclassic else np.isfinite(vals)&(vals>0)
        if coloured: valid &= np.isfinite(vals)
        rows,cols=np.nonzero(valid)
        x=(cols+.5)*cfg.bin_seconds/3600; y=rows+.5
        colours=cm(norm(vals[rows,cols])) if coloured else basecolour
        if cfg.style=='dots':
            ax.scatter(x if down else y,y if down else x,s=cfg.point_size**2,
                       c=colours,edgecolors='none',zorder=3,rasterized=False)
        elif cfg.style=='ticks':
            delta=cfg.mark_length*.43
            if down: ax.vlines(x,y-delta,y+delta,colors=colours,linewidth=cfg.line_width,zorder=3)
            else: ax.hlines(x,y-delta,y+delta,colors=colours,linewidth=cfg.line_width,zorder=3)
        elif cfg.style=='hticks':
            delta=cfg.mark_length*cfg.bin_seconds/7200
            if down: ax.hlines(y,x-delta,x+delta,colors=colours,linewidth=cfg.line_width,zorder=3)
            else: ax.vlines(y,x-delta,x+delta,colors=colours,linewidth=cfg.line_width,zorder=3)
        else:
            if cfg.bar_scale=='per_day':
                maxrows=np.max(np.where(np.isfinite(vals),vals,0),axis=1)
                divisor=np.maximum(maxrows[rows],1e-9)
            else: divisor=vmax
            fraction=np.minimum(vals[rows,cols]/divisor,1)
            if cfg.style=='barlength':
                delta=fraction*cfg.mark_length*cfg.bin_seconds/7200
                if down: ax.hlines(y,x-delta,x+delta,colors=colours,linewidth=cfg.line_width,zorder=3)
                else: ax.vlines(y,x-delta,x+delta,colors=colours,linewidth=cfg.line_width,zorder=3)
            else:
                base=rows+.90; tip=base-fraction*(.88*cfg.mark_length)
                if down: ax.vlines(x,base,tip,colors=colours,linewidth=cfg.line_width,zorder=3)
                else: ax.hlines(x,base,tip,colors=colours,linewidth=cfg.line_width,zorder=3)
    handles=draw_schedule(ax,analysis,p,cfg.style=='heatmap')
    handles+=draw_solar(ax,analysis,p)
    if coloured and cfg.show_colourbar:
        from matplotlib.cm import ScalarMappable
        mappable=ScalarMappable(norm=norm,cmap=cm)
        colourbar=fig.colorbar(mappable,ax=ax,fraction=.024,pad=.025,
                              extend='max' if np.any(vals>vmax) else 'neither')
        colourbar.set_label(('Mean per individual · ' if p.is_group and cfg.metric!='active_percent' else '')+p.unit,
                            fontsize=cfg.font_size-1,color=cfg.ink_colour)
        colourbar.ax.tick_params(labelsize=cfg.font_size-1,colors=cfg.ink_colour)
    if cfg.double_plot:
        if down: ax.axvline(24,color=cfg.ink_colour,linewidth=.8,alpha=.6,zorder=6)
        else: ax.axhline(24,color=cfg.ink_colour,linewidth=.8,alpha=.6,zorder=6)
    ticks=np.arange(0,hours+1e-8,cfg.tick_hours)
    if not len(ticks) or ticks[-1]<hours-1e-8: ticks=np.r_[ticks,hours]
    labels=[f'{h:g}' if cfg.alignment=='start24' else format_clock(int(round(h*3600+cfg.start_seconds))) for h in ticks]
    use_dates=cfg.label_mode=='dates' and p.origin is not None
    label_pitch=(cfg.font_size-1)/72*(1.5 if down else (3.6 if use_dates else 1.8))
    step=cfg.tick_days or max(1,int(math.ceil(days/32)),int(math.ceil(label_pitch/cfg.row_height)))
    di=list(range(0,days,step))
    if days-1 not in di:
        if not cfg.tick_days and len(di)>1 and days-1-di[-1]<step:
            di[-1]=days-1
        elif days-1-di[-1]>=step or len(di)==1 and days-1>=label_pitch/cfg.row_height:
            di.append(days-1)
    dlabels=[(p.origin+pd.Timedelta(days=int(analysis.row_indices[i]))).strftime('%d %b %Y')
             if use_dates else str(analysis.base_day+int(analysis.row_indices[i])+1) for i in di]
    time_title='Hours from recording start' if cfg.alignment=='start24' else 'Clock time'
    if cfg.double_plot: time_title+=' · day d | day d+1'
    day_title='Date' if use_dates else 'Recording day'
    if cfg.x_axis_title.strip(): x_title=cfg.x_axis_title.strip()
    else: x_title=time_title if down else day_title
    if cfg.y_axis_title.strip(): y_title=cfg.y_axis_title.strip()
    else: y_title=day_title if down else time_title
    if down:
        ax.set(xlim=(0,hours),ylim=(days,0),xticks=ticks,xticklabels=labels,
               yticks=np.array(di)+.5,yticklabels=dlabels,xlabel=x_title,ylabel=y_title)
    else:
        ax.set(xlim=(0,days),ylim=(hours,0),yticks=ticks,yticklabels=labels,
               xticks=np.array(di)+.5,xticklabels=dlabels,ylabel=y_title,xlabel=x_title)
        ax.tick_params(axis='x',labelrotation=65)
    ax.tick_params(labelsize=cfg.font_size-1,colors=cfg.ink_colour,length=cfg.tick_length,width=cfg.tick_width)
    ax.xaxis.label.set(fontsize=cfg.font_size,color=cfg.ink_colour)
    ax.yaxis.label.set(fontsize=cfg.font_size,color=cfg.ink_colour)
    for spine in ax.spines.values(): spine.set_color(cfg.ink_colour); spine.set_alpha(.3)
    if cfg.show_grid: ax.grid(axis='x' if down else 'y',color=cfg.ink_colour,alpha=.10,linewidth=.5)
    ax.set_title(cfg.title.strip() or p.name,loc='left',fontsize=cfg.font_size+4,color=cfg.ink_colour,pad=12)
    layout_actogram(fig,ax,colourbar,handles,cfg,days)
    return fig


def render_bytes(analysis: Analysis,key: str,fmt: str='png',dpi: int=125) -> tuple[bytes,tuple[float,float]]:
    if fmt not in {'png','svg','pdf'}: raise DataError('Unsupported image format.')
    with MPL_LOCK:
        fig=make_figure(analysis,key)
        try:
            w,h=fig.get_size_inches()
            if w*h*dpi*dpi>45_000_000:
                raise DataError('This image is too large at that resolution. Click this message away, go to Appearance and reduce export DPI/figure size or shorten the plotted range.')
            stream=io.BytesIO(); FigureCanvasAgg(fig)
            fig.savefig(stream,format=fmt,dpi=dpi,facecolor=fig.get_facecolor())
            return stream.getvalue(),(float(w),float(h))
        finally: fig.clear()


def panel_table(analysis: Analysis,key: str) -> pd.DataFrame:
    cfg,p=analysis.settings,analysis.panels[key]
    rows=analysis.row_indices; nb=DAY_SECONDS//cfg.bin_seconds
    di=np.repeat(rows,nb); bi=np.tile(np.arange(nb),len(rows))
    relative=di*DAY_SECONDS+bi*cfg.bin_seconds
    if p.origin is not None:
        times=pd.DatetimeIndex(p.origin+pd.to_timedelta(relative,unit='s'))
        timestamps=times.strftime('%Y-%m-%dT%H:%M:%S')
        schedule=classify_schedule(times,np.full(len(times),cfg.bin_seconds),cfg)
    else:
        timestamps=['']*len(di)
        schedule=pd.DataFrame({'treatment':['No common clock']*len(di),'phase':['No common clock']*len(di),
                               'illumination':['']*len(di),'lux':[np.nan]*len(di)})
    result=pd.DataFrame({
        'panel':p.name,'kind':'group' if p.is_group else 'individual','group':p.group,
        'recording_day':di+analysis.base_day+1,'bin_start_iso':timestamps,
        'bin_seconds':cfg.bin_seconds,'activity_seconds':p.seconds[rows].ravel(),
        'observed_seconds':p.observed[rows].ravel(),'coverage_percent':p.observed[rows].ravel()/cfg.bin_seconds*100,
        'episode_starts':p.episodes[rows].ravel(),'plotted_intensity':p.values[rows].ravel(),
        'unfiltered_intensity':(p.unfiltered_values if p.unfiltered_values is not None else p.values)[rows].ravel(),
        'intensity_units':p.unit,'threshold_active':p.binary[rows].ravel(),
        'activity_threshold':cfg.threshold,'threshold_units':p.threshold_units,
        'threshold_rule':'>=' if cfg.comparison=='ge' else '>',
        'active_subjects_percent':p.active_percent[rows].ravel(),
        'n_intensity_contributors':p.n[rows].ravel(),'n_threshold_contributors':p.n_binary[rows].ravel(),
        'bin_assignment':'Whole source intervals by start',
        'schedule_assignment':'Display-bin timestamps; crossings marked Boundary interval',
    })
    return pd.concat([result,schedule],axis=1)


def activity_statistics(analysis: Analysis) -> pd.DataFrame:
    out=[]
    for key,p in analysis.panels.items():
        if p.is_group: continue
        df=panel_table(analysis,key)
        valid=df['threshold_active'].notna()&(df['observed_seconds']>0)
        df=df.loc[valid]
        for (period,phase),sub in df.groupby(['treatment','phase'],sort=False,dropna=False):
            obs=float(sub.observed_seconds.sum())
            total=float(sub.activity_seconds.sum()) if sub.activity_seconds.notna().any() else np.nan
            out.append({'id':key.removeprefix('subject:'),'label':p.name,'group':p.group,
                'condition':f'{period} | {phase}','treatment':period,'phase':phase,
                'total_activity_seconds':total,'observed_seconds':obs,
                'active_percent':total/obs*100 if obs>0 else np.nan,
                'mean_plotted_intensity':float(sub.plotted_intensity.mean()),'intensity_units':p.unit,
                'observed_bins':len(sub),'observed_days':sub.recording_day.nunique(),
                'phase_assignment':'Display-bin start; boundary-crossing bins are a separate condition'})
    return pd.DataFrame(out)


def group_statistics(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty: return pd.DataFrame()
    columns=['total_activity_seconds','observed_seconds','active_percent','mean_plotted_intensity','observed_bins','observed_days']
    return table.groupby(['group','condition'],sort=True)[columns].agg(['mean','median','count']).reset_index().pipe(
        lambda f:f.set_axis(['_'.join(c).strip('_') if isinstance(c,tuple) else c for c in f.columns],axis=1))


def render_comparison(table: pd.DataFrame,cfg: Settings,a: str,b: str,measure: str,
                      fmt: str='png',dpi: int=125) -> tuple[bytes,tuple[float,float]]:
    labels={'active_percent':'Time active / observed time (%)','total_activity_seconds':'Recorded activity (hours)',
            'mean_plotted_intensity':'Mean plotted intensity (imported/display units)'}
    if measure not in labels or a==b: raise DataError('Choose two different conditions and a supported measure.')
    left=table.loc[table.condition==a,['id','label','group',measure]].rename(columns={measure:'a'})
    right=table.loc[table.condition==b,['id',measure]].rename(columns={measure:'b'})
    df=left.merge(right,on='id',how='outer')
    # Fill labels/groups for subjects with data only in condition B.
    lookup=table.drop_duplicates('id').set_index('id')
    for c in ('label','group'): df[c]=df[c].fillna(df.id.map(lookup[c]))
    if df.empty or not np.isfinite(df[['a','b']].to_numpy(float)).any():
        raise DataError('No usable data for that comparison. Activity seconds/percent require duration measurements.')
    factor=3600 if measure=='total_activity_seconds' else 1
    with MPL_LOCK:
        fig=Figure(figsize=(max(7,min(15,3+df.group.nunique()*2.1)),5.6),facecolor=cfg.background_colour)
        ax=fig.add_subplot(111); ax.set_facecolor(cfg.background_colour)
        for i,(group,sub) in enumerate(df.groupby('group',sort=True)):
            jitter=np.linspace(-.06,.06,len(sub)) if len(sub)>1 else np.array([0.])
            vals=sub[['a','b']].to_numpy(float)/factor
            for j,(va,vb) in enumerate(vals):
                x1,x2=i-.19+jitter[j],i+.19+jitter[j]
                if np.isfinite(va) and np.isfinite(vb):
                    ax.plot([x1,x2],[va,vb],color=cfg.ink_colour,alpha=.25,linewidth=.7)
                for x,v,c in ((x1,va,cfg.light_colour),(x2,vb,cfg.dark_colour)):
                    if np.isfinite(v):
                        ax.scatter(x,v,s=26,color=c,zorder=3)
                        ax.annotate(str(sub.iloc[j]['label']),(x,v),xytext=(0,5),textcoords='offset points',ha='center',fontsize=7,color=cfg.ink_colour)
            for k,c in enumerate((cfg.light_colour,cfg.dark_colour)):
                v=vals[:,k]; v=v[np.isfinite(v)]
                if len(v): ax.scatter(i+(-.19 if k==0 else .19),np.median(v),marker='D',s=65,color=c,edgecolors=cfg.ink_colour,linewidth=.6,zorder=4)
        ax.scatter([],[],color=cfg.light_colour,label=a)
        ax.scatter([],[],color=cfg.dark_colour,label=b)
        ax.scatter([],[],marker='D',s=65,facecolors='none',edgecolors=cfg.ink_colour,label='Group median')
        groups=sorted(df.group.unique())
        ax.set_xticks(range(len(groups)),groups)
        ax.set_ylabel(labels[measure]); ax.set_title(cfg.title or 'Activity comparison',loc='left')
        ax.legend(frameon=False,fontsize=8)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        fig.tight_layout()
        try:
            stream=io.BytesIO(); FigureCanvasAgg(fig)
            fig.savefig(stream,format=fmt,dpi=dpi,facecolor=fig.get_facecolor())
            return stream.getvalue(),tuple(fig.get_size_inches())
        finally: fig.clear()


def batch_zip(analysis: Analysis,kind: str,fmt: str) -> bytes:
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('settings.json',json.dumps(asdict(analysis.settings),ensure_ascii=False,indent=2))
        stats=activity_statistics(analysis)
        z.writestr('individual_summaries.csv',csv_bytes(stats))
        z.writestr('group_summaries.csv',csv_bytes(group_statistics(stats)))
        z.writestr('INTERPRETATION.txt','\n'.join(analysis.notes))
        for i,(key,p) in enumerate(analysis.panels.items(),1):
            if (kind=='subjects' and p.is_group) or (kind=='groups' and not p.is_group): continue
            stem=f'{i:03d}_{safe_filename(key)}'
            raw,_=render_bytes(analysis,key,fmt,analysis.settings.export_dpi)
            z.writestr(f'plots/{stem}.{fmt}',raw)
            z.writestr(f'tables/{stem}.csv',csv_bytes(panel_table(analysis,key)))
    return stream.getvalue()


def save_project_bytes(subjects: list[Subject],cfg: Settings,groups: dict[str,str],opts: ImportOptions) -> bytes:
    stream=io.BytesIO()
    meta={'schema':3,'app_version':VERSION,'settings':asdict(cfg),'import_options':asdict(opts),
          'group_colours':groups,'subjects':[]}
    with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as z:
        for i,s in enumerate(subjects):
            path=f'data/{i:04d}.csv'; df=s.data.copy(); df['time']=df.time.map(lambda t:t.isoformat())
            payload=csv_bytes(df); z.writestr(path,payload)
            meta['subjects'].append({'id':s.id,'label':s.label,'group':s.group,'colour':s.colour,
                                    'included':s.included,'sources':s.sources,'measure_kind':s.measure_kind,
                                    'value_unit':s.value_unit,'aggregation':s.aggregation,'data_file':path,
                                    'data_sha256':hashlib.sha256(payload).hexdigest()})
        z.writestr('project.json',json.dumps(meta,ensure_ascii=False,indent=2))
    return stream.getvalue()


def load_project_bytes(raw: bytes) -> tuple[list[Subject],Settings,dict[str,str],ImportOptions]:
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            if sum(i.file_size for i in z.infolist())>MAX_IMPORT_BYTES:
                raise DataError('This project exceeds the 256 MB uncompressed limit.')
            meta=json.loads(z.read('project.json'))
            if meta.get('schema')!=3:
                raise DataError('This project was created with an incompatible version of Actogram Lab.')
            cfg=Settings(**{k:v for k,v in meta['settings'].items() if k in {f.name for f in fields(Settings)}})
            if cfg.heatmap_schedule=='none': cfg.shade_mode='none'
            cfg.heatmap_schedule='shade'
            cfg.validate()
            opts=ImportOptions(**{k:v for k,v in meta.get('import_options',{}).items() if k in {f.name for f in fields(ImportOptions)}})
            groups={str(k):str(v) for k,v in meta['group_colours'].items()}
            for c in groups.values(): to_rgb(c)
            subjects=[]; ids=set()
            for item in meta['subjects']:
                if item['id'] in ids: raise DataError('Duplicate individual IDs in project.')
                ids.add(item['id']); path=item['data_file']
                if not re.fullmatch(r'data/\d{4,}\.csv',path): raise DataError('Unexpected project data path.')
                payload=z.read(path)
                if hashlib.sha256(payload).hexdigest()!=item['data_sha256']: raise DataError('Project data checksum mismatch.')
                frame=pd.read_csv(io.BytesIO(payload)); frame['time']=pd.to_datetime(frame.time,format='mixed',errors='raise')
                kind=item.get('measure_kind','duration')
                if kind not in {'duration','numeric','events'}: raise DataError('Unsupported project measurement type.')
                frame,_=validate_frame(frame,item['id'],kind)
                if item.get('colour'): to_rgb(item['colour'])
                subjects.append(Subject(item['id'],item['group'],frame,item.get('label',''),item.get('colour'),
                                        bool(item.get('included',True)),item.get('sources',[]),kind,
                                        item.get('value_unit','Activity seconds'),item.get('aggregation','sum')))
            return subjects,cfg,groups,opts
    except (zipfile.BadZipFile,KeyError,TypeError,json.JSONDecodeError) as exc:
        raise DataError(f'Invalid project archive: {exc}') from exc

# Flet interface adjust the theme and such
class ActogramApp:
    ACCENT="#11D1EB"
    BACKGROUND='#F5F4F0'
    SURFACE='#FFFFFF'
    TEXT='#24272C'
    MUTED='#70757C'
    BORDER='#E4E4DF'

    def __init__(self,page):
        self.page=page
        self.subjects=[]
        self.group_colours={'Ungrouped':NAMED_COLOURS['Ink']}
        self.cfg=Settings(); self.import_options=ImportOptions()
        self.analysis=None; self.current_key=''; self.dirty=True; self.busy=False
        self.controls={}; self.colour_buttons={}; self.logs=[]; self.section='data'
        self.view_kind='actogram'; self.stats_snapshot=None
        self.image=None; self.preview_png=None; self.preview_inches=(11.4,6.0)
        self.zoom_factor=1.0; self.preview_scroll=None
        self.dynamic_panels={}
        page.title='Actogram Lab'
        page.theme=ft.Theme(color_scheme_seed=self.ACCENT)
        page.theme_mode=ft.ThemeMode.LIGHT
        page.bgcolor=self.BACKGROUND; page.padding=16; page.spacing=12
        if not page.web:
            page.window.width=1440; page.window.height=940
            page.window.min_width=1040; page.window.min_height=720
        page.on_resize=self.on_resize
        self.build_controls(); self.build_layout(); self.refresh_data()

    def text(self,value,size=12,bold=False,colour=None):
        return ft.Text(value,size=size,weight=ft.FontWeight.BOLD if bold else ft.FontWeight.NORMAL,
                       color=colour or self.TEXT)

    def dropdown(self,label,options,value,callback=None,**kwargs):
        return ft.Dropdown(label=label,value=value,options=[ft.DropdownOption(key=k,text=v) for k,v in options.items()],
                           on_select=callback,text_size=13,**kwargs)

    def note(self,message):
        return ft.Container(content=self.text(message,11,colour=self.MUTED),padding=10,
                            bgcolor='#F7F6F2',border_radius=8)

    def heading(self,title,detail=None):
        controls=[self.text(title,14,True)]
        if detail: controls.append(self.text(detail,11,colour=self.MUTED))
        return ft.Column(controls,spacing=4)

    def build_controls(self):
        def tf(key,label,hint=None):
            value=getattr(self.cfg,key)
            c=ft.TextField(label=label,value='' if value is None else str(value),hint_text=hint,
                           text_size=13,on_change=self.mark_dirty)
            self.controls[key]=c
        def dd(key,label,options):
            self.controls[key]=self.dropdown(label,options,getattr(self.cfg,key),self.mark_dirty)
        def sw(key,label):
            self.controls[key]=ft.Switch(label=label,value=getattr(self.cfg,key),on_change=self.mark_dirty)
        def sl(key,lo,hi,divisions):
            self.controls[key]=ft.Slider(min=lo,max=hi,divisions=divisions,value=getattr(self.cfg,key),label='{value}',on_change=self.mark_dirty)
        dd('style','Plot style',STYLES)
        self.style_hint=self.note(STYLE_HINTS[self.cfg.style])
        self.day_spacing_label=self.text(f'Day spacing · {self.cfg.row_height:.2f} inches')
        tf('bin_minutes','Display bin (minutes)','1, 5, 10, 15, 30, 60…')
        dd('metric','Activity measure',METRICS)
        tf('window_start','Plot from · date and time','Blank = first available interval')
        tf('window_end','Plot until · date and time','Blank = last available interval')
        sw('double_plot','Double plot · 48 hours')
        tf('threshold','Activity threshold','0 keeps any positive activity')
        dd('comparison','Threshold rule',{'ge':'At least (≥)','gt':'More than (>)'})
        self.threshold_hint=self.note('')
        sw('schedule_enabled','Show light / dark phases')
        dd('simple_cycle','illumination pattern',CYCLES)
        tf('lights_on','Lights on','06:00')
        tf('lights_off','Lights off','18:00')
        dd('cmap','Activity colour palette',CMAPS)
        sw('reverse_cmap','Reverse the palette')
        sw('show_colourbar','Show activity colour scale')
        dd('scale_mode','Intensity scale',{'shared':'Shared between plots','auto':'Automatic for each plot','manual':'Fixed maximum'})
        tf('vmax','Fixed intensity maximum')
        dd('mark_colour_mode','Colour of activity marks',{'solid':'Individual / fixed colour','intensity':'Intensity colour palette'})
        sw('use_group_colours','Use individual / group colours')
        dd('bar_scale','Bar length / height scale',{'global':'Use selected intensity scale','per_day':'Scale each row separately'})
        dd('orientation','Orientation',{'days_down':'Days down · time across','days_across':'Days across · time down'})
        dd('shade_mode','Highlight these phases',{'both':'Light and dark','light':'Light only','dark':'Dark only','none':'Neither'})
        dd('label_mode','Day labels',{'dates':'Calendar dates','days':'Recording-day numbers'})
        tf('figure_width','Figure width (inches)')
        tf('tick_hours','Time-axis ticks every (hours)')
        tf('tick_days','Day labels every (days)','0 = automatic')
        tf('title','Plot title','Blank = individual / group name')
        tf('x_axis_title','X-axis title','Blank = automatic')
        tf('y_axis_title','Y-axis title','Blank = automatic')
        tf('export_dpi','Export resolution (dpi)')
        sw('show_grid','Show time grid')
        sl('point_size',.5,10,38)
        sl('line_width',.1,5,49)
        sl('mark_length',.05,1,19)
        sl('row_height',.06,1.2,114)
        self.controls['row_height'].label='{value} in'
        sl('font_size',6,22,16)
        sl('tick_length',0,12,24)
        sl('tick_width',.1,3,29)
        sl('shade_opacity',0,1,50)
        sw('show_schedule_legend','Show phase / treatment legend')
        sw('solar_enabled','Overlay solar elevation / twilight curves')
        for key,label in [('solar_horizon','Horizon · 0°'),('solar_civil','Civil twilight · −6°'),
                          ('solar_nautical','Nautical twilight · −12°'),('solar_astronomical','Astronomical twilight · −18°')]:
            sw(key,label)
            colour_key=key+'_colour'
            self.colour_buttons[colour_key]=self.colour_button(label,getattr(self.cfg,colour_key),
                lambda e,k=colour_key,l=label:self.colour_dialog(l,getattr(self.cfg,k),lambda colour:self.set_colour(k,colour)))
        tf('solar_city','Field location / city','For example Tromsø, Norway')
        tf('solar_latitude','Latitude','Decimal degrees')
        tf('solar_longitude','Longitude','Decimal degrees')
        tf('solar_timezone','Time zone','Europe/Oslo')
        dd('solar_time_basis','Clock basis of imported data',{'local':'Local civil clock (DST included)',
            'standard':'Fixed standard clock (no DST)','utc':'UTC clock'})
        for key,label in {'light_colour':'Light phase','dark_colour':'Dark phase','low_colour':'Gradient low',
                          'high_colour':'Gradient high','background_colour':'Plot background',
                          'missing_colour':'Missing data','ink_colour':'Text / fixed marks'}.items():
            self.colour_buttons[key]=self.colour_button(label,getattr(self.cfg,key),
                lambda e,k=key,l=label:self.colour_dialog(l,getattr(self.cfg,k),lambda colour:self.set_colour(k,colour)))
        self.subject_cards=ft.Column(spacing=6)
        self.group_cards=ft.Column(spacing=6)
        self.data_summary=self.text('No recordings imported',12,colour=self.MUTED)
        self.recording_range=self.text('Dates and times appear here after importing.',11,colour=self.MUTED)
        self.group_name=ft.TextField(label='Group name',hint_text='For example Control or North',text_size=13)
        self.status=self.text('Import recordings to begin.',12,colour=self.MUTED)
        self.summary=self.text('',11,colour=self.MUTED)
        self.progress=ft.ProgressBar(visible=False,color=self.ACCENT,height=3)
        self.target=self.dropdown('Individual or group',{},'',self.target_changed,expand=True)
        self.render_button=ft.Button('Update plot',icon=ft.Icons.REFRESH,on_click=self.render,
                                     style=ft.ButtonStyle(bgcolor=self.ACCENT,color="#1B1B1A"))
        self.zoom_label=self.text('Fit',11,colour=self.MUTED)

    def colour_button(self,label,colour,callback):
        return ft.Container(content=ft.Row([ft.Container(width=23,height=23,bgcolor=colour,border_radius=6,
                                    border=ft.Border.all(1,self.BORDER)),self.text(label)],spacing=10),
                            padding=9,border=ft.Border.all(1,self.BORDER),border_radius=8,on_click=callback)

    def build_layout(self):
        brand=ft.Row([ft.Container(ft.Icon(ft.Icons.BLUR_ON_SHARP,color="#272525",size=30),
                                   padding=2,bgcolor=self.ACCENT,border_radius=12),
                      self.text('Actogram Lab',24,True)],spacing=12) 
        # add subtext for BiRBS Lab Innovations
        brand.controls.append(self.text('BiRBS Lab Innovations',12,colour=self.MUTED))
    

        
        self.header_actions=ft.Row([
            ft.Button('Add files',icon=ft.Icons.ADD,on_click=self.pick_csvs),
            ft.IconButton(icon=ft.Icons.FOLDER_OPEN_OUTLINED,tooltip='Open project',on_click=self.open_project),
            ft.IconButton(icon=ft.Icons.SAVE_OUTLINED,tooltip='Save project',on_click=self.save_project),
            ft.IconButton(icon=ft.Icons.HELP_OUTLINE,tooltip='Help',on_click=self.show_help)],spacing=5)
        header=ft.Row([brand,self.header_actions],alignment=ft.MainAxisAlignment.SPACE_BETWEEN)
        self.nav_buttons={}
        for key,label in [('data','Data'),('plot','Plot'),('appearance','Appearance')]:
            self.nav_buttons[key]=ft.TextButton(label,on_click=lambda e,k=key:self.navigate(k))
        self.left_body=ft.Container(expand=True)
        self.sidebar=ft.Container(width=370,padding=14,bgcolor=self.SURFACE,border_radius=15,
            border=ft.Border.all(1,self.BORDER),content=ft.Column([
                ft.Row(list(self.nav_buttons.values()),spacing=4),ft.Divider(height=1),self.left_body],expand=True,spacing=12))
        self.preview_host=ft.Container(expand=True,padding=6,bgcolor='#ECEDE9',border_radius=10,content=self.empty_preview())
        zooms=ft.Row([
            ft.IconButton(icon=ft.Icons.ZOOM_OUT,tooltip='Zoom out',on_click=self.zoom_out),
            ft.IconButton(icon=ft.Icons.ZOOM_IN,tooltip='Zoom in',on_click=self.zoom_in),
            ft.IconButton(icon=ft.Icons.FIT_SCREEN,tooltip='Fit whole plot',on_click=self.reset_view),
            self.zoom_label],spacing=0)
        self.exports=ft.Row([
            ft.OutlinedButton('PNG',on_click=self.export_png),ft.OutlinedButton('SVG',on_click=self.export_svg),
            ft.OutlinedButton('PDF',on_click=self.export_pdf),ft.OutlinedButton('Table',on_click=self.export_csv),
            ft.OutlinedButton('Batch ZIP',on_click=self.batch_dialog),
            ft.TextButton('Summaries',on_click=self.summary_dialog)],spacing=5,wrap=True)
        workspace=ft.Container(expand=True,padding=14,bgcolor=self.SURFACE,border_radius=15,
            border=ft.Border.all(1,self.BORDER),content=ft.Column([
                ft.Row([self.target,self.render_button],spacing=10),
                ft.Row([self.status],wrap=True),self.progress,self.preview_host,
                ft.Row([zooms,self.text('Right-click plot to copy · scroll to explore',11,colour=self.MUTED)],
                       alignment=ft.MainAxisAlignment.SPACE_BETWEEN,wrap=True),
                self.exports,self.summary],expand=True,spacing=9))
        footer=ft.Column([self.text('Developed by Hamid Taghipour',11,True,colour=self.MUTED),
                          self.text('UiT The Arctic University of Norway',10,colour=self.MUTED)],
                         spacing=2,horizontal_alignment=ft.CrossAxisAlignment.CENTER)
        self.page.add(ft.Column([header,ft.Row([self.sidebar,workspace],expand=True,spacing=14),footer],expand=True,spacing=10))
        self.navigate('data')

    def empty_preview(self):
        return ft.Container(content=ft.Column([
            ft.Icon(ft.Icons.BAR_CHART,size=56,color=self.ACCENT),self.text('Waiting for your command!',23,True),
            self.text('Add files, configure plotting and view the results here.',13,colour=self.MUTED)
            ],alignment=ft.MainAxisAlignment.CENTER,horizontal_alignment=ft.CrossAxisAlignment.CENTER,spacing=15),
            expand=True,alignment=ft.Alignment.CENTER,padding=25)

    def navigate(self,section):
        self.section=section; c=self.controls; self.dynamic_panels={}
        for key,btn in self.nav_buttons.items():
            btn.style=ft.ButtonStyle(color=self.ACCENT if key==section else self.MUTED,
                                    bgcolor='#FFF0E5' if key==section else '#FFFFFF')
        if section=='data':
            content=[self.heading('Recordings',''),
                ft.Row([ft.Button('Add files',icon=ft.Icons.UPLOAD_FILE,on_click=self.pick_csvs),
                        ft.TextButton('Load folder',on_click=self.pick_folder)],wrap=True),
                ft.TextButton('Convert TXT-CSV',on_click=self.pick_text_conversion),
                ft.TextButton('Special import',on_click=self.pick_mapped),self.data_summary,
                self.recording_range,
                ft.Row([ft.TextButton('Include all',on_click=self.include_all),
                        ft.TextButton('Exclude all',on_click=self.exclude_all),
                        ft.IconButton(icon=ft.Icons.DELETE_OUTLINE,tooltip='Remove included individuals',on_click=self.remove_included)],spacing=0),
                self.subject_cards,ft.Divider(),self.heading('Groups & individual colours'),self.group_cards,
                self.group_name,ft.OutlinedButton('Assign included individuals to this group',on_click=self.assign_group),
                self.note('Click a name to rename it or change its group. Click a colour swatch to change that individual’s plotting colour.'),
                ft.TextButton('Import / analysis log',on_click=self.show_log)]
        elif section=='plot':
            self.dynamic_panels['threshold']=ft.Column([c['threshold'],c['comparison'],self.threshold_hint],spacing=10)
            self.dynamic_panels['measure']=ft.Column([c['metric']],spacing=10)
            self.dynamic_panels['simple_times']=ft.Row([c['lights_on'],c['lights_off']],spacing=10)
            c['lights_on'].expand=True; c['lights_off'].expand=True
            self.dynamic_panels['simple_schedule']=ft.Column([
                c['simple_cycle'],self.dynamic_panels['simple_times'],
                ft.Row([self.colour_buttons['light_colour'],self.colour_buttons['dark_colour']],wrap=True)],spacing=10)
            self.dynamic_panels['multiple_schedule']=ft.Column([
                self.note(self.treatment_summary()),ft.TextButton('Return to one basic cycle',on_click=self.simple_schedule_clicked)],spacing=7)
            self.dynamic_panels['schedule']=ft.Column([
                self.dynamic_panels['simple_schedule'],self.dynamic_panels['multiple_schedule'],
                ft.OutlinedButton('Advanced treatments…',icon=ft.Icons.SETTINGS_OUTLINED,on_click=self.treatments_dialog)],spacing=10)
            content=[self.heading('Plot'),c['style'],self.style_hint,
                self.day_spacing_label,c['row_height'],
                self.text('Move left to bring days closer. ',11,colour=self.MUTED),c['bin_minutes'],
                ft.TextButton('Use source bin size',on_click=self.use_source_bin),self.dynamic_panels['measure'],
                self.dynamic_panels['threshold'],c['double_plot'],ft.Divider(),
                self.heading('Date / time range','DD/MM/YYYY HH:MM or YYYY-MM-DD HH:MM:SS'),
                self.recording_range,c['window_start'],c['window_end'],
                ft.Row([ft.TextButton('Use full range',on_click=self.full_range),
                        ft.TextButton('Trim / exclude…',on_click=self.range_dialog)],wrap=True),
                self.note('Start is included; end is excluded.'),
                ft.Divider(),self.heading('Light / dark schedule'),c['schedule_enabled'],self.dynamic_panels['schedule'],
                ft.Divider(),self.heading('Outdoor solar / twilight curves'),ft.OutlinedButton('Configure solar curves…',icon=ft.Icons.WB_SUNNY_OUTLINED,on_click=self.solar_dialog),ft.Divider(),ft.TextButton('Advanced plotting…',on_click=self.advanced_plot_dialog)]
        else:
            self.dynamic_panels['dot']=ft.Column([self.text('Dot diameter (points)'),c['point_size']],spacing=3)
            self.dynamic_panels['line']=ft.Column([self.text('Tick / bar thickness (points)'),c['line_width'],
                self.text('Maximum mark length / bar height'),c['mark_length']],spacing=3)
            self.dynamic_panels['barscale']=ft.Column([c['bar_scale']],spacing=5)
            self.dynamic_panels['markcolours']=ft.Column([c['mark_colour_mode'],c['use_group_colours']],spacing=6)
            self.dynamic_panels['gradient_custom']=ft.Row([self.colour_buttons['low_colour'],self.colour_buttons['high_colour']],wrap=True)
            self.dynamic_panels['palette']=ft.Column([c['cmap'],ft.TextButton('Browse palettes…',on_click=self.palette_gallery),
                c['reverse_cmap'],self.dynamic_panels['gradient_custom'],c['show_colourbar']],spacing=7)
            self.dynamic_panels['scale']=ft.Column([c['scale_mode'],c['vmax']],spacing=9)
            self.dynamic_panels['phase_shading']=ft.Column([c['shade_mode'],self.text('Phase shading opacity'),c['shade_opacity']],spacing=6)
            content=[self.heading('Appearance','Only settings for the selected plot style are shown.'),c['style'],self.style_hint,
                self.dynamic_panels['dot'],self.dynamic_panels['line'],self.dynamic_panels['barscale'],
                self.dynamic_panels['markcolours'],self.dynamic_panels['palette'],self.dynamic_panels['scale'],
                ft.Divider(),self.heading('Light / dark shading'),self.dynamic_panels['phase_shading'],c['show_schedule_legend'],
                self.note('Phase colours are set in Plot, or separately for each period under Advanced treatments.'),
                ft.Divider(),self.heading('Figure & axes'),c['orientation'],c['figure_width'],
                self.day_spacing_label,c['row_height'],
                self.text('Move left to bring days closer.',11,colour=self.MUTED),
                self.text('Font size'),c['font_size'],c['tick_hours'],c['tick_days'],
                self.text('Axis tick length'),c['tick_length'],self.text('Axis tick thickness'),c['tick_width'],
                c['label_mode'],c['show_grid'],c['x_axis_title'],c['y_axis_title'],self.colour_buttons['background_colour'],
                self.colour_buttons['ink_colour'],self.colour_buttons['missing_colour'],c['title'],c['export_dpi']]
        self.left_body.content=ft.Column(content,scroll=ft.ScrollMode.AUTO,expand=True,spacing=11)
        self.sync_visibility(); self.page.update()

    def sync_visibility(self):
        c=self.controls; style=c['style'].value; heat=style=='heatmap'
        self.style_hint.content.value=STYLE_HINTS.get(style,'')
        if not heat and c['mark_colour_mode'].value=='intensity':
            self.style_hint.content.value+=' Colour also represents the selected activity measure.'
        self.day_spacing_label.value=f'Day spacing · {float(c["row_height"].value):.2f} inches'
        def show(key,value):
            if key in self.dynamic_panels: self.dynamic_panels[key].visible=bool(value)
        show('threshold',True)
        show('measure',True)
        selected=next((s for s in self.subjects if s.included),None)
        units=threshold_units(selected)
        c['threshold'].label=f'Activity threshold · {units}'
        self.threshold_hint.content.value='Values that fall outside the rule become zero; missing data stay missing.'
        if c['metric'].value=='active_percent':
            self.threshold_hint.content.value+=f' Individuals are classified using {units}, then their percentages are calculated.'
        elif c['metric'].value in {'minutes','percent','episodes'}:
            self.threshold_hint.content.value+=f' The threshold still uses {units} before displaying minutes, percentages or episode counts.'
        else:
            self.threshold_hint.content.value+=' '
        show('schedule',c['schedule_enabled'].value)
        show('simple_schedule',self.cfg.schedule_mode=='simple')
        show('multiple_schedule',self.cfg.schedule_mode=='multiple')
        show('simple_times',c['simple_cycle'].value=='LD')
        show('dot',style=='dots'); show('line',style in {'ticks','hticks','bars','barlength'})
        show('barscale',style in {'bars','barlength'})
        show('markcolours',not heat)
        palette=heat or c['mark_colour_mode'].value=='intensity'
        show('palette',palette); show('gradient_custom',c['cmap'].value=='custom')
        show('scale',palette or style in {'bars','barlength'})
        c['vmax'].visible=c['scale_mode'].value=='manual'
        c['figure_width'].visible=c['orientation'].value=='days_down'
        show('phase_shading',c['schedule_enabled'].value)

    def mark_dirty(self,_=None):
        self.dirty=True
        self.status.value='Settings changed · select Update plot'; self.status.color=self.ACCENT
        self.sync_visibility(); self.page.update()

    def read_settings(self):
        cfg=replace(self.cfg,treatments=[dict(t) for t in self.cfg.treatments])
        ints={'tick_days','export_dpi'}
        defaults=Settings()  # Stable types even when JSON numbers were encoded as integers.
        for key,control in self.controls.items():
            value=control.value; default=getattr(defaults,key)
            try:
                if isinstance(default,bool): value=bool(value)
                elif key=='vmax': value=float(value) if str(value or '').strip() else None
                elif key in ints:
                    v=float(value)
                    if not v.is_integer(): raise ValueError()
                    value=int(v)
                elif isinstance(default,float): value=float(value)
                else: value=str(value or '').strip()
            except (TypeError,ValueError,OverflowError) as exc:
                raise DataError(f'Enter a valid number for {key.replace("_"," ")}.') from exc
            setattr(cfg,key,value)
        cfg.validate(); return cfg

    def load_settings_controls(self):
        if self.cfg.heatmap_schedule=='none': self.cfg.shade_mode='none'
        self.cfg.heatmap_schedule='shade'
        for key,control in self.controls.items():
            value=getattr(self.cfg,key)
            control.value=('' if value is None else str(value)) if isinstance(control,ft.TextField) else value
        for key,button in self.colour_buttons.items():
            button.content.controls[0].bgcolor=getattr(self.cfg,key)
        self.sync_visibility()

    def treatment_summary(self):
        if not self.cfg.treatments: return 'No treatment periods defined.'
        rows=[]
        for item in self.cfg.treatments:
            rows.append(f'{item.get("name","Period")}: {item.get("cycle","LD")} · {item.get("start","")} → {item.get("end","")}')
        return f'{len(rows)} treatment period(s)\n'+'\n'.join(rows)

    def set_colour(self,key,colour):
        setattr(self.cfg,key,colour)
        self.colour_buttons[key].content.controls[0].bgcolor=colour
        self.mark_dirty()

    def data_bounds(self):
        selected=[s for s in self.subjects if s.included] or self.subjects
        if not selected: return None
        shift=pd.Timedelta(minutes=self.cfg.timezone_offset_min)
        return (min(s.data.time.min() for s in selected)+shift,
                max((s.data.time+pd.to_timedelta(s.data.duration,unit='s')).max() for s in selected)+shift)

    def refresh_data(self):
        self.data_summary.value=f'{len(self.subjects)} individuals · {sum(s.included for s in self.subjects)} included · {sum(len(s.data) for s in self.subjects):,} rows'
        bounds=self.data_bounds()
        self.recording_range.value=f'Available: {datetime_text(bounds[0])} → {datetime_text(bounds[1])}' if bounds else 'No recording dates available yet.'
        cards=[]
        for s in self.subjects:
            check=ft.Checkbox(value=s.included,on_change=lambda e,sid=s.id:self.set_included(sid,e.control.value))
            colour=s.colour or self.group_colours.get(s.group,NAMED_COLOURS['Ink'])
            kind='source intervals' if s.measure_kind=='duration' else s.value_unit.lower()
            cards.append(ft.Container(padding=7,border_radius=9,bgcolor='#FAFAF7',content=ft.Row([
                check,ft.Container(content=ft.Column([self.text(s.name,13,True),
                    self.text(f'{s.group} · {len(s.data):,} {kind}',10,colour=self.MUTED),
                    self.text(f'{nominal_source_seconds(s):g}-second source interval',10,colour=self.MUTED)],spacing=2),
                    expand=True,on_click=lambda e,sid=s.id:self.edit_subject(sid)),
                ft.Container(width=25,height=25,bgcolor=colour,border_radius=6,tooltip='Individual colour',
                    border=ft.Border.all(1,self.BORDER),on_click=lambda e,sid=s.id:self.subject_colour_dialog(sid))],spacing=2)))
        self.subject_cards.controls=cards or [self.note('Add CSV, TSV or delimited TXT files to begin.')]
        groups=sorted({s.group for s in self.subjects})
        self.group_cards.controls=[self.colour_button(f'{g} · {sum(s.group==g for s in self.subjects)} individuals',
            self.group_colours.get(g,NAMED_COLOURS['Ink']),
            lambda e,g=g:self.colour_dialog(f'Group · {g}',self.group_colours.get(g,NAMED_COLOURS['Ink']),lambda colour:self.set_group_colour(g,colour))) for g in groups]
        options={f'subject:{s.id}':f'Individual · {s.name}' for s in self.subjects if s.included}
        options.update({f'group:{g}':f'Group · {g}' for g in sorted({s.group for s in self.subjects if s.included})})
        self.target.options=[ft.DropdownOption(key=k,text=v) for k,v in options.items()]
        if self.target.value not in options: self.target.value=next(iter(options),'')
        self.page.update()

    def full_range(self,_=None):
        self.controls['window_start'].value=''; self.controls['window_end'].value=''
        self.cfg.trim_first_hours=self.cfg.trim_last_hours=0.; self.cfg.excluded_dates=''
        self.mark_dirty()

    def range_dialog(self,_=None):
        first=ft.TextField(label='Ignore first N hours of each recording',value=str(self.cfg.trim_first_hours))
        last=ft.TextField(label='Ignore last N hours of each recording',value=str(self.cfg.trim_last_hours))
        dates=ft.TextField(label='Exclude whole calendar dates',value=self.cfg.excluded_dates,hint_text='22/09/2026, 25/09/2026',multiline=True)
        def preset(field,value): field.value=str(value); self.page.update()
        def exclude_edge(which):
            selected=[s for s in self.subjects if s.included]
            if not selected:
                self.error('Include recordings first.'); return
            existing=parse_excluded_dates(str(dates.value or ''))
            shift=pd.Timedelta(minutes=self.cfg.timezone_offset_min)
            for subject in selected:
                if which=='first': edge=subject.data.time.min()+shift
                else:
                    edge=(subject.data.time+pd.to_timedelta(subject.data.duration,unit='s')).max()+shift-pd.Timedelta(nanoseconds=1)
                existing.add(edge.normalize())
            dates.value=', '.join(d.strftime('%d/%m/%Y') for d in sorted(existing)); self.page.update()
        def apply(_):
            try:
                cfg=self.read_settings(); cfg.trim_first_hours=float(first.value); cfg.trim_last_hours=float(last.value)
                cfg.excluded_dates=str(dates.value or '').strip(); cfg.validate()
            except Exception as exc: self.error(str(exc)); return
            self.cfg=cfg; self.page.pop_dialog(); self.mark_dirty()
        self.page.show_dialog(ft.AlertDialog(title=self.text('Trim / exclude recording time',20,True),
            content=ft.Column([self.note('Trims are relative to each individual’s full recording and also respect the plotting start/end. 24 hours means a rolling 24-hour interval, not a calendar date.'),
                first,ft.TextButton('Ignore first 2 hours',on_click=lambda e:preset(first,2)),
                last,ft.TextButton('Ignore last 24 hours',on_click=lambda e:preset(last,24)),dates,
                ft.Row([ft.TextButton('Exclude first calendar date(s)',on_click=lambda e:exclude_edge('first')),
                        ft.TextButton('Exclude last calendar date(s)',on_click=lambda e:exclude_edge('last'))],wrap=True),
                self.note('Excluded dates apply to every included individual. Edge-date shortcuts add the first/last dates found across included recordings. Middle dates remain visible as missing rows; time is not compressed.')],
                width=560,height=450,scroll=ft.ScrollMode.AUTO,spacing=12),
            actions=[ft.TextButton('Cancel',on_click=self.close_dialog),ft.Button('Apply',on_click=apply)]))

    def advanced_plot_dialog(self,_=None):
        cfg=self.cfg
        align=self.dropdown('Align days',{'clock24':'Actual clock times (recommended)','start24':'Elapsed from each recording’s start'},cfg.alignment)
        day=ft.TextField(label='Each clock-aligned row starts at',value=cfg.day_start,hint_text='00:00')
        crop=self.dropdown('Crop source intervals',{'complete':'Only complete intervals inside the plotting range',
                           'start':'Whole intervals whose START is inside the range'},cfg.crop_policy)
        placement=self.dropdown('Assign source rows to display bins',{'start':'Whole row by start timestamp (normal)',
                                'strict':'Require every source interval to fit a display bin'},cfg.bin_policy)
        coverage=ft.TextField(label='Minimum observed coverage per display bin (%)',value=str(cfg.minimum_coverage))
        minimum=ft.TextField(label='Minimum individuals contributing to a group bin',value=str(cfg.minimum_subjects))
        active=ft.TextField(label='Group classic marks: active individuals required (%)',value=str(cfg.group_active_percent))
        offset=ft.TextField(label='Explicit clock correction (minutes)',value=str(cfg.timezone_offset_min))
        def apply(_):
            try:
                new=self.read_settings(); new.alignment=align.value; new.day_start=str(day.value)
                new.crop_policy=crop.value; new.bin_policy=placement.value
                new.minimum_coverage=float(coverage.value); new.group_active_percent=float(active.value)
                m=float(minimum.value); o=float(offset.value)
                if not m.is_integer() or not o.is_integer(): raise DataError('Minimum individuals and clock offset must be whole numbers.')
                new.minimum_subjects=int(m); new.timezone_offset_min=int(o); new.validate()
            except Exception as exc: self.error(str(exc)); return
            self.cfg=new; self.page.pop_dialog(); self.refresh_data(); self.mark_dirty()
        self.page.show_dialog(ft.AlertDialog(title=self.text('Advanced plotting',20,True),
            content=ft.Column([align,day,crop,placement,coverage,minimum,active,offset,
                self.note('Clock correction shifts imported timestamps, not the schedule you enter. Default 0 preserves file times.'),
                self.note('')],
                width=600,height=560,scroll=ft.ScrollMode.AUTO,spacing=13),
            actions=[ft.TextButton('Cancel',on_click=self.close_dialog),ft.Button('Apply',on_click=apply)]))

    def solar_dialog(self,_=None):
        c=self.controls
        status=self.text('Enter a city and resolve it, or type latitude and longitude manually.',11,colour=self.MUTED)
        def resolve(_=None):
            city=str(c['solar_city'].value or '').strip()
            if not city:
                self.error('Enter a city name, e.g. Tromsø, Norway.'); return
            async def work():
                try:
                    from geopy.geocoders import Nominatim
                    from timezonefinder import TimezoneFinder
                    def lookup():
                        place=Nominatim(user_agent='actogram-Lab-solar/3.5',timeout=12).geocode(city,exactly_one=True)
                        if place is None: raise DataError('City not found. Try City, Country or enter coordinates manually.')
                        tz=TimezoneFinder().timezone_at(lat=place.latitude,lng=place.longitude)
                        return place.latitude,place.longitude,tz,place.address
                    lat,lon,tz,address=await asyncio.to_thread(lookup)
                    c['solar_latitude'].value=f'{lat:.6f}'; c['solar_longitude'].value=f'{lon:.6f}'
                    if tz: c['solar_timezone'].value=tz
                    status.value=f'Resolved: {address} · {lat:.5f}, {lon:.5f} · {tz or "set timezone manually"}'
                    self.page.update()
                except ImportError:
                    self.error('Automatic city lookup requires: pip install geopy timezonefinder. Coordinates can be entered manually without these packages.')
                except Exception as exc: self.error(str(exc))
            self.page.run_task(work)
        def apply(_=None):
            try:
                new=self.read_settings()
                new.solar_enabled=bool(c['solar_enabled'].value)
                new.validate()
            except Exception as exc: self.error(str(exc)); return
            self.cfg=new; self.page.pop_dialog(); self.mark_dirty()
        items=[c['solar_enabled'],c['solar_city'],ft.OutlinedButton('Find city coordinates automatically',on_click=resolve),
               ft.Row([c['solar_latitude'],c['solar_longitude']],spacing=8),c['solar_timezone'],c['solar_time_basis'],
               status,self.note('The selected clock basis must match the imported timestamps. Fixed standard time removes seasonal DST jumps from the solar overlay, but does not automatically shift imported activity timestamps.'),
               self.heading('Solar elevation curves')]
        for key,label,ls,colour in SOLAR_LEVELS:
            items.append(ft.Row([c[key],self.colour_buttons[key+'_colour']],wrap=True))
        items.append(self.note('Polar day/night may have no crossing of a given elevation; those curves are omitted on those days.'))
        self.page.show_dialog(ft.AlertDialog(title=self.text('Outdoor solar curves',20,True),
            content=ft.Column(items,width=650,height=570,scroll=ft.ScrollMode.AUTO,spacing=10),
            actions=[ft.TextButton('Cancel',on_click=self.close_dialog),ft.Button('Apply',on_click=apply)]))

    def simple_schedule_clicked(self,_=None):
        self.cfg.schedule_mode='simple'; self.navigate('plot'); self.mark_dirty()

    def treatments_dialog(self,_=None):
        entries=[]
        cards=ft.Column(spacing=14)
        count=ft.TextField(label='How many treatment periods?',value=str(len(self.cfg.treatments) or 1),width=245,text_size=13)
        period_days=ft.TextField(label='Days per period',value='7',width=180,text_size=13)
        first_start=ft.TextField(label='First period starts',value='',hint_text='13/04/2026 06:00',width=245,text_size=13)
        def build_entry(t,index):
            state={'light_colour':t.light_colour,'dark_colour':t.dark_colour}
            controls={
                'name':ft.TextField(label='Period name',value=t.name,text_size=13),
                'start':ft.TextField(label='Starts · date and time',value=t.start,hint_text='29/09/2026 06:00',text_size=13,expand=True),
                'end':ft.TextField(label='Ends · date and time (exclusive)',value=t.end,hint_text='06/10/2026 06:00',text_size=13,expand=True),
                'cycle':self.dropdown('Light-Dark pattern',CYCLES,t.cycle),
                'lights_on':ft.TextField(label='Lights on each day',value=t.lights_on,expand=True,text_size=13),
                'lights_off':ft.TextField(label='Lights off each day',value=t.lights_off,expand=True,text_size=13),
                'light_name':ft.TextField(label='Light phase: illumination',value=t.light_name,hint_text='White light / Red light',expand=True,text_size=13),
                'light_lux':ft.TextField(label='Light-phase lux (optional)',value='' if t.light_lux is None else str(t.light_lux),expand=True,text_size=13),
                'dark_name':ft.TextField(label='Dark phase: illumination',value=t.dark_name,expand=True,text_size=13),
                'dark_lux':ft.TextField(label='Dark-phase lux (optional)',value='' if t.dark_lux is None else str(t.dark_lux),expand=True,text_size=13),
            }
            swatches={}
            def choose(which):
                def selected(colour):
                    state[which]=colour; swatches[which].content.controls[0].bgcolor=colour; self.page.update()
                self.colour_dialog('Period highlight colour',state[which],selected)
            for key,label in [('light_colour','Light highlight'),('dark_colour','Dark highlight')]:
                swatches[key]=self.colour_button(label,state[key],lambda e,k=key:choose(k))
            times=ft.Row([controls['lights_on'],controls['lights_off']],spacing=12)
            light=ft.Column([ft.Row([controls['light_name'],controls['light_lux']],spacing=12),swatches['light_colour']],spacing=9)
            dark=ft.Column([ft.Row([controls['dark_name'],controls['dark_lux']],spacing=12),swatches['dark_colour']],spacing=9)
            def sync(_=None):
                mode=controls['cycle'].value
                times.visible=mode=='LD'; light.visible=mode in {'LD','LL'}; dark.visible=mode in {'LD','DD'}
                self.page.update()
            controls['cycle'].on_select=sync
            times.visible=t.cycle=='LD'; light.visible=t.cycle in {'LD','LL'}; dark.visible=t.cycle in {'LD','DD'}
            card=ft.Container(content=ft.Column([self.text(f'Period {index+1}',14,True),controls['name'],
                ft.Row([controls['start'],controls['end']],spacing=12),controls['cycle'],times,light,dark],spacing=12),
                padding=15,border=ft.Border.all(1,self.BORDER),border_radius=10)
            return {'controls':controls,'state':state,'card':card}
        initial=[Treatment(**t) for t in self.cfg.treatments]
        if not initial:
            bounds=self.data_bounds()
            initial=[Treatment(start=datetime_text(bounds[0]) if bounds else '',end=datetime_text(bounds[1]) if bounds else '',
                lights_on=str(self.controls['lights_on'].value or '06:00'),lights_off=str(self.controls['lights_off'].value or '18:00'),
                cycle=self.controls['simple_cycle'].value,light_colour=self.cfg.light_colour,dark_colour=self.cfg.dark_colour)]
        entries.extend(build_entry(t,i) for i,t in enumerate(initial))
        def refresh():
            cards.controls=[entry['card'] for entry in entries]; count.value=str(len(entries)); self.page.update()
        def change_count(_=None,delta=None):
            try:
                value=float(count.value) if delta is None else len(entries)+delta
                if not float(value).is_integer() or not 1<=value<=24:
                    raise DataError('Choose between 1 and 24 treatment periods.')
                n=int(value)
            except Exception as exc: self.error(str(exc)); return
            # No silent loss of entered periods when reducing the count.
            def resize():
                while len(entries)<n:
                    previous=entries[-1]
                    t=Treatment(name=f'Period {len(entries)+1}',start=str(previous['controls']['end'].value or ''),end='',
                        cycle=previous['controls']['cycle'].value,
                        lights_on=str(previous['controls']['lights_on'].value),lights_off=str(previous['controls']['lights_off'].value),
                        light_name=str(previous['controls']['light_name'].value),dark_name=str(previous['controls']['dark_name'].value),
                        light_colour=previous['state']['light_colour'],dark_colour=previous['state']['dark_colour'])
                    entries.append(build_entry(t,len(entries)))
                del entries[n:]
                refresh()
            if n<len(entries): self.confirm('Remove the final period(s) from this editor? Their entered settings will be discarded.',resize)
            else: resize()
        def auto_periods(_=None):
            try:
                n=float(count.value); days=float(period_days.value)
                if not n.is_integer() or not 1<=n<=24: raise DataError('Choose between 1 and 24 treatment periods.')
                if not days.is_integer() or days<1: raise DataError('Days per period must be a positive whole number.')
                n=int(n); days=int(days)
                if str(first_start.value or '').strip(): start=parse_datetime(first_start.value)
                else:
                    bounds=self.data_bounds()
                    if not bounds: raise DataError('Enter the first period start date/time or import data first.')
                    start=bounds[0].normalize()+pd.Timedelta(seconds=parse_clock(str(self.controls['lights_on'].value or '06:00')))
                    first_start.value=datetime_text(start)
                template=entries[0] if entries else None
                entries.clear()
                for i in range(n):
                    a=start+pd.Timedelta(days=i*days); b=a+pd.Timedelta(days=days)
                    c=template['controls']; st=template['state']
                    def lv(key):
                        raw=str(c[key].value or '').strip(); return float(raw) if raw else None
                    t=Treatment(name=f'Period {i+1}',start=datetime_text(a),end=datetime_text(b),cycle=c['cycle'].value,
                        lights_on=str(c['lights_on'].value),lights_off=str(c['lights_off'].value),light_name=str(c['light_name'].value),
                        light_lux=lv('light_lux'),dark_name=str(c['dark_name'].value),dark_lux=lv('dark_lux'),
                        light_colour=st['light_colour'],dark_colour=st['dark_colour'])
                    entries.append(build_entry(t,i))
                refresh()
            except Exception as exc: self.error(str(exc))
        def apply(_):
            try:
                periods=[]
                for entry in entries:
                    data={key:str(control.value or '').strip() for key,control in entry['controls'].items()}
                    for key in ('light_lux','dark_lux'): data[key]=float(data[key]) if data[key] else None
                    t=Treatment(**data,**entry['state']); t.validate(True)
                    t.start=datetime_text(parse_datetime(t.start)); t.end=datetime_text(parse_datetime(t.end))
                    periods.append(asdict(t))
                # Read the main settings using the current mode first, then validate the new protocol.
                cfg=self.read_settings(); cfg.schedule_mode='multiple'; cfg.schedule_enabled=True; cfg.treatments=periods
                cfg.treatments=[asdict(t) for t in validated_treatments(cfg)]; cfg.validate()
            except Exception as exc: self.error(str(exc)); return
            self.cfg=cfg; self.controls['schedule_enabled'].value=True
            self.page.pop_dialog(); self.navigate('plot'); self.mark_dirty()
        cards.controls=[entry['card'] for entry in entries]
        self.page.show_dialog(ft.AlertDialog(title=self.text('Advanced treatments',21,True),
            content=ft.Column([
                self.note('Define the actual chronological periods of the experiment. This one protocol applies to all included recordings. Dates here describe treatments '),
                ft.Row([count,ft.OutlinedButton('Set number',on_click=change_count),
                        ft.TextButton('+ Period',on_click=lambda e:change_count(delta=1))],wrap=True),
                ft.Row([first_start,period_days,ft.OutlinedButton('Make equal periods',on_click=auto_periods)],wrap=True),
                self.note('Equal periods are consecutive. The first starts at the entered light-phase onset; each next period starts exactly N days later at the same clock time.'),
                self.note('Example: baseline white LD → red 50 lux LD → red 20 lux LD → constant darkness.'),
                cards,self.note('End times are exclusive. One period can end exactly when the next starts. Overlaps are rejected; gaps are left unassigned.')],
                width=800,height=590,scroll=ft.ScrollMode.AUTO,spacing=13),
            actions=[ft.TextButton('Cancel',on_click=self.close_dialog),ft.Button('Use these periods',on_click=apply)]))

    async def pick_text_conversion(self,_=None):
        if self.busy: return
        try:
            payloads=await self.pick_payloads(['txt','tsv','csv','log','dat'],True)
            if payloads: await self.text_conversion_dialog(payloads)
        except Exception as exc: self.error(str(exc))

    async def text_conversion_dialog(self,payloads):
        if not payloads: return
        try: suggested=suggest_text_options(payloads[0][1])
        except DataError: suggested=TextImportOptions()
        known_records=suggested.header and suggested.delimiter=='tab' and suggested.pad_short_rows
        state={'prepared':None,'views':None,'signature':None,'working':False}
        choices={'auto':'Automatic','tab':'Tabs','whitespace':'Spaces / whitespace',
                 'semicolon':'Semicolons','comma':'Commas','pipe':'Vertical bars |',
                 'fixed':'Fixed column widths','custom':'Another separator'}
        separator=self.dropdown('Columns are separated by',choices,
            suggested.delimiter if suggested.delimiter in choices else 'custom',expand=True)
        separator_hint=self.note('With whitespace separation, an unquoted date and clock time become separate columns. Use date, time and activity headings, or quote a combined timestamp.')
        separator_hint.visible=separator.value=='whitespace'
        custom=ft.TextField(label='Separator character',value='',text_size=13,visible=False)
        widths=ft.TextField(label='Column widths',value=suggested.widths,
            hint_text='Characters per column, for example 10, 9, 5',text_size=13,visible=False)
        skipped=ft.TextField(label='Skip first lines',value=str(suggested.skip_rows),width=145,text_size=13)
        header=ft.Checkbox(label='First remaining row contains column names',value=suggested.header)
        names=ft.TextField(label='Column names (optional)',value=suggested.column_names,
            hint_text='For example date, time, activity',text_size=13)
        encoding=self.dropdown('Text encoding',{'auto':'Automatic','utf-8-sig':'UTF-8',
            'utf-16':'UTF-16 with byte-order mark','utf-16-le':'UTF-16 little endian',
            'utf-16-be':'UTF-16 big endian','cp1252':'Windows-1252'},'auto')
        padding=ft.Checkbox(label='Keep unequal row lengths: fill blank cells and retain extra columns',
            value=suggested.pad_short_rows)
        record=ft.TextField(label='Keep only this first-column value (optional)',value='',
            hint_text='Blank keeps every row; enter D to keep only D records',text_size=13)
        record_note=self.note('This export contains different record types. Keep every row to save the complete CSV, or select the timestamped record type you want to import. The converter preserves repeated records and does not decide which field measures activity.')
        record_note.visible=known_records
        more=ft.Column([names,encoding],spacing=10,visible=False)
        records=ft.Column([padding,record],spacing=10,visible=known_records)
        selected=self.dropdown('Preview file',{str(i):name for i,(name,_) in enumerate(payloads)},'0')
        selected.visible=len(payloads)>1
        summary=self.text('Choose the text layout, then preview the converted table.',12)
        report=self.note(''); report.visible=False
        original=ft.Text('',size=11,font_family='monospace',selectable=True,no_wrap=True)
        original_box=ft.Container(content=ft.Row([original],scroll=ft.ScrollMode.AUTO),
            bgcolor='#F7F6F2',padding=8,visible=False)
        table=ft.Container(content=self.text('The CSV preview will appear here.',12,colour=self.MUTED),
            bgcolor='#F7F6F2',padding=8)
        def option_values():
            try:
                count=int(str(skipped.value or '0').strip())
                if count<0: raise ValueError
            except (TypeError,ValueError) as exc:
                raise DataError('Enter a nonnegative whole number of lines to skip.') from exc
            return TextImportOptions(delimiter=str(custom.value or '') if separator.value=='custom' else separator.value,
                skip_rows=count,header=bool(header.value),column_names=str(names.value or ''),
                encoding=encoding.value,widths=str(widths.value or ''),pad_short_rows=bool(padding.value),
                record_type=str(record.value or '').strip())
        def changed(_=None):
            custom.visible=separator.value=='custom'; widths.visible=separator.value=='fixed'
            separator_hint.visible=separator.value=='whitespace'
            state['prepared']=state['views']=state['signature']=None
            save_button.disabled=import_button.disabled=True
            table.content=self.text('Settings changed. Preview again before saving or importing.',12,colour=self.MUTED)
            summary.value='Preview the table to check these settings.'; summary.color=self.MUTED
            report.visible=False
            self.page.update()
        def toggle_more(_=None):
            more.visible=not more.visible
            if not known_records: records.visible=more.visible
            self.page.update()
        def toggle_original(_=None):
            original_box.visible=not original_box.visible
            self.page.update()
        def show_preview(_=None):
            if not state['views']: return
            view=state['views'][int(selected.value or '0')]
            frame=view['preview']; meta=view['meta']; original.value=view['original']
            summary.value=f'{view["rows"]:,} rows × {len(frame.columns)} columns · {view["name"]}'
            summary.color=self.TEXT
            messages=[f'{meta["source_rows"]:,} source rows; {meta["filtered_rows"]:,} excluded by the record filter; {meta["skipped_lines"]} initial lines skipped.']
            if meta['padded_rows']: messages.append(f'{meta["padded_rows"]:,} shorter records have blank cells for fields they do not contain.')
            if meta['renamed_columns']:
                messages.append('Repeated or blank headings were made distinct: '+', '.join(item['to'] for item in meta['renamed_columns'])+'.')
            if meta['added_columns']: messages.append('Columns without source headings: '+', '.join(meta['added_columns'])+'.')
            if meta['blank_lines']: messages.append(f'{meta["blank_lines"]} empty lines were ignored.')
            report.content.value='\n'.join(messages); report.visible=True
            table.content=ft.Row([ft.DataTable(
                columns=[ft.DataColumn(self.text(str(col),11,True)) for col in frame.columns],
                rows=[ft.DataRow(cells=[ft.DataCell(self.text(str(value),11)) for value in row])
                      for row in frame.itertuples(index=False,name=None)],column_spacing=18)],scroll=ft.ScrollMode.AUTO)
            self.page.update()
        async def preview(_=None):
            if state['working'] or self.busy: return False
            try: options=option_values()
            except DataError as exc:
                summary.value=str(exc); summary.color=NAMED_COLOURS['Red']; self.page.update(); return False
            state['working']=True; body.disabled=True
            for button in buttons: button.disabled=True
            self.set_busy(True,'Preparing the CSV preview…')
            def prepare():
                prepared=[]; views=[]; total=0; used=set()
                for name,raw in payloads:
                    try: frame=text_table(raw,options)
                    except Exception as exc: raise DataError(f'{name}: {exc}') from exc
                    converted=csv_bytes(frame); total+=len(converted)
                    if total>MAX_IMPORT_BYTES: raise DataError('The converted batch exceeds 256 MB. Convert fewer files at a time.')
                    stem=safe_filename(Path(name).stem)
                    suffix='_'+safe_filename(options.record_type) if options.record_type else ''
                    output_name=f'{stem}{suffix}_converted.csv'; number=2
                    while output_name.casefold() in used:
                        output_name=f'{stem}{suffix}_{number}_converted.csv'; number+=1
                    used.add(output_name.casefold()); prepared.append((output_name,converted))
                    views.append({'name':output_name,'preview':frame.head(8),'rows':len(frame),'meta':dict(frame.attrs),
                                  'original':text_preview_lines(raw,options.encoding)})
                return prepared,views
            success=False
            try:
                state['prepared'],state['views']=await asyncio.to_thread(prepare)
                state['signature']=asdict(options); show_preview(); success=True
            except Exception as exc:
                state['prepared']=state['views']=state['signature']=None
                summary.value=str(exc); summary.color=NAMED_COLOURS['Red']; report.visible=False
                table.content=self.text('Adjust the text settings and preview again. No data have been imported.',12,colour=self.MUTED)
                try: original.value=text_preview_lines(payloads[int(selected.value or '0')][1],encoding.value)
                except DataError as failure: original.value=str(failure)
                original_box.visible=True
            finally:
                state['working']=False; body.disabled=False
                for button in buttons: button.disabled=False
                save_button.disabled=import_button.disabled=not success
                self.set_busy(False)
            return success
        async def save(_=None):
            if state['working'] or self.busy or not state['prepared']: return
            try:
                if asdict(option_values())!=state['signature']: changed(); return
                state['working']=True; body.disabled=True
                for button in buttons: button.disabled=True
                self.set_busy(True,'Saving converted CSV data…')
                if len(state['prepared'])==1:
                    name,raw=state['prepared'][0]
                    await self.save_bytes(raw,name,'csv')
                else:
                    def archive():
                        buffer=io.BytesIO()
                        with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_DEFLATED) as bundle:
                            for name,raw in state['prepared']: bundle.writestr(name,raw)
                        return buffer.getvalue()
                    await self.save_bytes(await asyncio.to_thread(archive),'Converted_CSV_files.zip','zip')
            except Exception as exc: self.error(str(exc))
            finally:
                state['working']=False; body.disabled=False
                for button in buttons: button.disabled=False
                save_button.disabled=import_button.disabled=not bool(state['prepared'])
                self.set_busy(False)
        async def import_converted(_=None):
            if state['working'] or self.busy or not state['prepared']: return
            try:
                if asdict(option_values())!=state['signature']: changed(); return
                prepared=list(state['prepared'])
                self.page.pop_dialog()
                await self.route_import(prepared,allow_conversion=False)
            except Exception as exc: self.error(str(exc))
        separator.on_select=encoding.on_select=changed
        for control in (custom,widths,skipped,names,record,header,padding): control.on_change=changed
        selected.on_select=show_preview
        body=ft.Column([
            self.note(f'{len(payloads)} file(s). Check the table before saving a CSV or continuing to import. Dates and activity values stay as written; surrounding field padding is trimmed.'),
            selected,separator,separator_hint,custom,widths,ft.Row([skipped,header],wrap=True),
            record_note,records,ft.TextButton('More text options…',on_click=toggle_more),more,
            ft.TextButton('Show / hide original text',on_click=toggle_original),original_box,
            summary,table,report],width=850,height=565,scroll=ft.ScrollMode.AUTO,spacing=11)
        cancel_button=ft.TextButton('Cancel',on_click=self.close_dialog)
        preview_button=ft.OutlinedButton('Preview conversion',on_click=preview)
        save_button=ft.OutlinedButton('Save CSV' if len(payloads)==1 else 'Save CSVs as ZIP',on_click=save,disabled=True)
        import_button=ft.Button('Use for import',on_click=import_converted,disabled=True)
        buttons=[cancel_button,preview_button,save_button,import_button]
        self.page.show_dialog(ft.AlertDialog(title=self.text('Convert text to CSV',21,True),content=body,
            modal=True,actions=buttons))
        await preview()


    async def route_import(self,payloads,allow_conversion=True,force_mapping=False):
        if not payloads: return
        def columns(raw):
            try:
                text,_=decode_text_file(raw[:65536])
                try: sep=csv.Sniffer().sniff(text,delimiters=',;\t|').delimiter
                except csv.Error: sep='\t' if '\t' in text.splitlines()[0] else ','
                return [normal_name(value) for value in next(csv.reader(io.StringIO(text),delimiter=sep))]
            except Exception: return []
        headers=[columns(raw) for _,raw in payloads]
        if allow_conversion and any(header[:3]==['f','date','hmsec'] for header in headers):
            await self.text_conversion_dialog(payloads)
            return
        if not force_mapping and all({'bin_start_iso','motion_seconds'}.issubset(header) for header in headers):
            await self.import_payloads(payloads,ImportOptions(subject=NONE,date_order='auto'))
            return
        if not force_mapping and all({'iso_time','t_since_start_s','motion_flag'}.issubset(h) for h in headers):
            await self.import_payloads(payloads,ImportOptions(timestamp='iso_time',activity='motion_flag',subject='snail_id',
                data_mode='samples',units='binary',aggregation='activity',sample_seconds=2.,raw_bin_minutes=1.,
                date_order='ymd'),display_bin_minutes=1.)
            return
        if not force_mapping and all(any(h.startswith('date_time') for h in header) and
                any(h.startswith('occupancy') for h in header) for header in headers):
            await self.import_payloads(payloads,ImportOptions(),display_bin_minutes=1.)
            return
        try:
            frame=await asyncio.to_thread(read_table,payloads[0][1])
        except DataError:
            if not allow_conversion: raise
            await self.text_conversion_dialog(payloads)
            return
        if (allow_conversion and len(frame.columns)==1
                and normal_name(frame.columns[0]) not in set(TIMESTAMP_COLUMNS)|{'date','recorded_at','time_stamp','elapsed_seconds','elapsed_minutes'}):
            await self.text_conversion_dialog(payloads)
            return
        self.import_dialog(payloads,list(frame.columns),frame.head(5))

    async def pick_mapped(self,_=None):
        if self.busy: return
        try:
            payloads=await self.pick_payloads(['csv','tsv','txt'],True)
            if payloads: await self.route_import(payloads,force_mapping=True)
        except Exception as exc: self.error(str(exc))

    def import_dialog(self,payloads,columns,sample):
        opts=self.import_options
        reuse=opts.timestamp in columns
        lookup={normal_name(c):c for c in columns}
        raw_schema={'iso_time','t_since_start_s','motion_flag'}.issubset(lookup)
        def guess(aliases,default=''):
            return next((lookup[a] for a in aliases if a in lookup),default)
        def column(label,value,aliases=(),optional=False):
            choices={NONE:'Use the filename for each individual'} if optional else {}
            choices.update({col:col for col in columns})
            selected=value if value in choices else guess(aliases,NONE if optional else '')
            return self.dropdown(label,choices,selected,expand=True)
        time_choice=opts.timestamp_kind if reuse else 'calendar'
        separate=guess(('date','recording_date','day'))
        combined=guess(('bin_start_iso','timestamp_local','timestamp','datetime','date_time','timestamp_iso','timestamp_utc','iso_time'))
        if (reuse and opts.date_column in columns) or (not reuse and separate and not combined and 'time' in lookup):
            time_choice='split'
        time_format=self.dropdown('How is time stored?',{
            'calendar':'Date and time in one column',
            'split':'Date and time in separate columns',
            'elapsed_s':'Seconds from recording start',
            'elapsed_min':'Minutes from recording start',
            'unix_s':'Unix timestamp in seconds (UTC)',
            'unix_ms':'Unix timestamp in milliseconds (UTC)'},time_choice)
        timestamp=column('Timestamp column',opts.timestamp if reuse else '',
            ('bin_start_iso','timestamp_local','timestamp','datetime','date_time','timestamp_iso','timestamp_utc','iso_time','time','elapsed_seconds','elapsed_minutes','elapsed','bin_start_sec','t_since_start_s'))
        date=column('Date column',opts.date_column if reuse else '',('date','recording_date','day'))
        date_choice=opts.date_order if reuse and opts.date_order else ('dmy' if opts.day_first else 'mdy') if reuse else 'auto'
        date_order=self.dropdown('Date order',{
            'auto':'Automatic · read the file values',
            'dmy':'Day / month / year · 20/10/2026',
            'mdy':'Month / day / year · 10/20/2026',
            'ymd':'Year / month / day · 2026-10-20',
            'ydm':'Year / day / month · 2026-20-10'},date_choice)
        timestamp_note=self.note('')
        raw_note=self.note('Raw logger ticks: precise elapsed time keeps repeated clock seconds distinct. Flags count active records by default. Gaps over 1.5 times the normal sample interval remain unknown after the first interval.')
        start=ft.TextField(label='Recording start · date and time',value=opts.recording_start if reuse else '',
            hint_text='2026-10-22 06:00:00',text_size=13)
        end=ft.TextField(label='Recording end · date and time',value=opts.recording_end if reuse else '',
            hint_text='2026-10-29 06:00:00',text_size=13)
        activity=column('Activity column',opts.activity if reuse else '',
            ('motion_seconds','activity_seconds','active_seconds','motion_minutes','activity_minutes','motion_flag','activity','active','motion','state','flag','count','counts','value'))
        measure_choice='seconds'
        if reuse:
            if opts.data_mode=='events': measure_choice='events'
            elif opts.units in {'binary','binary_flag'}: measure_choice='flag'
            elif opts.aggregation!='activity': measure_choice='counts'
            elif opts.units=='minutes': measure_choice='minutes'
        elif normal_name(activity.value or '') in {'motion_minutes','activity_minutes'}:
            measure_choice='minutes'
        elif raw_schema and activity.value==lookup['motion_flag']:
            measure_choice='flag'
        measure=self.dropdown('How is activity recorded?',{
            'seconds':'Seconds active in each row',
            'minutes':'Minutes active in each row',
            'flag':'0 / 1 or inactive / active',
            'counts':'Activity counts in each row',
            'events':'A row is logged only when activity occurs'},measure_choice)
        flag_meaning=self.dropdown('What does an active flag mean?',{
            'binary_flag':'Activity occurred sometime during this interval',
            'binary':'Active for the entire interval'},
            'binary' if reuse and opts.units=='binary' else 'binary_flag')
        row_seconds=opts.sample_seconds if reuse and opts.data_mode=='samples' else (opts.fallback_seconds if reuse else (1. if raw_schema else 60.))
        row_unit='minutes' if row_seconds>=60 and abs(row_seconds/60-round(row_seconds/60))<1e-7 else 'seconds'
        interval=ft.TextField(label='Each row covers',value=f'{row_seconds/(60 if row_unit=="minutes" else 1):g}',text_size=13,expand=True)
        interval_unit=self.dropdown('Interval unit',{'seconds':'Seconds','minutes':'Minutes'},row_unit,expand=True)
        interval_row=ft.Row([interval,interval_unit],spacing=12)
        bins=ft.TextField(label='Plot bin size (minutes)',value=f'{opts.raw_bin_minutes if reuse else 1:g}',
            hint_text='1, 5, 10, 15, 30, 60',text_size=13)
        continuous=ft.Checkbox(label='The logger was running continuously between these times',value=False)
        activity_note=self.note('')
        preview_text=self.text('',11,colour=self.MUTED)
        subject=column('Individual ID column (optional)',opts.subject if reuse else NONE,optional=True)
        decimal=self.dropdown('Decimal separator',{'.':'Point (12.5)',',':'Comma (12,5)'},opts.decimal if reuse else '.')
        clock=self.dropdown('Timestamp time zone',{'wall':'Use the written clock time','utc':'Convert time-zone offsets to UTC'},
            opts.time_mode if reuse else 'wall')
        optional_fields=ft.Column([subject,decimal,clock],spacing=10,visible=False)
        preview=ft.Container(content=ft.Row([ft.DataTable(
            columns=[ft.DataColumn(self.text(str(col),11,True)) for col in columns],
            rows=[ft.DataRow(cells=[ft.DataCell(self.text('' if pd.isna(value) else str(value),11)) for value in row])
                  for row in sample.itertuples(index=False,name=None)],column_spacing=18)],scroll=ft.ScrollMode.AUTO),
            bgcolor='#F7F6F2',padding=8,visible=False)
        def toggle(control):
            control.visible=not control.visible
            self.page.update()
        def sync(_=None,update=True):
            events=measure.value=='events'
            calendar=time_format.value in {'calendar','split'}
            raw_selected=raw_schema and time_format.value=='calendar' and timestamp.value==lookup['iso_time'] and not events
            raw_note.visible=raw_selected
            interval.label='Normal sample interval' if raw_selected else 'Each row covers'
            date.visible=time_format.value=='split'
            timestamp.label='Time column' if date.visible else 'Timestamp column'
            date_order.visible=calendar
            timestamp_note.visible=calendar
            clock.visible=calendar
            start.visible=events or time_format.value in {'elapsed_s','elapsed_min'}
            end.visible=continuous.visible=events
            activity.visible=interval_row.visible=not events
            flag_meaning.visible=measure.value=='flag'
            if events:
                message='Each row counts as one event. Empty bins count as zero only within the continuous recording period entered below. The result is events per bin.'
            elif measure.value=='flag' and flag_meaning.value=='binary_flag':
                message='Active rows are counted in each plot bin. A flag does not specify how many seconds the individual was active.'
            elif measure.value=='flag' and raw_selected:
                message='An active state is held until the next sample. If a gap exceeds 1.5 times the normal sample interval, only one normal interval is counted and the rest remains unknown.'
            elif measure.value=='flag':
                message='An active row contributes its full interval as active time. For example, an active 1-second row adds 1 second. Missing rows remain unknown.'
            elif measure.value=='counts':
                message='Counts are added within each plot bin. Missing rows remain unknown. Use zero for an observed interval with no activity.'
            else:
                message='Activity durations are added within each plot bin. Each timestamp marks the start of its row interval. Missing rows remain unknown.'
            activity_note.content.value=message
            if calendar:
                timestamp_note.content.value='Date order applies to the text stored in the file. Spreadsheet display formats can differ. ISO T, fractional seconds and time-zone suffixes are accepted.'
                timestamp_note.content.color=self.MUTED
            if not sample.empty and timestamp.value in sample.columns:
                row=sample.iloc[0]
                raw_times=sample[timestamp.value].astype(str)
                if date.visible and date.value in sample.columns: raw_times=sample[date.value].astype(str)+' '+raw_times
                time=str(raw_times.iloc[0])
                value='1 event' if events else (str(row[activity.value]) if activity.value in sample.columns else 'choose an activity column')
                preview_text.value=f'First row activity: {value}'
                if calendar:
                    timestamp_note.content.value+=f'\nFile value (unchanged): {time!r}'
                    try:
                        parsed=parse_times(raw_times,clock.value,date_order.value in {'dmy','ydm'},date_order.value).iloc[0]
                        timestamp_note.content.value+='\nRead as: '+parsed.strftime('%d %b %Y %H:%M:%S')+(' UTC' if clock.value=='utc' else '')+'.'
                        if date_order.value=='auto':
                            inferred=automatic_date_order(raw_times)
                            label={'ymd':'year / month / day','ydm':'year / day / month','mdy':'month / day / year','dmy':'day / month / year'}[inferred]
                            timestamp_note.content.value+=' Automatic recognised '+label+'.'
                        timestamp_note.content.color=self.MUTED
                    except (DataError,ValueError,TypeError) as exc:
                        timestamp_note.content.value+='\n'+str(exc)
                        if date_order.value=='auto':
                            timestamp_note.content.value+=' Import checks the full column; choose an explicit order if the dates remain ambiguous.'
                        elif re.match(r'^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}(?:[Tt\s]|$)',time) and not str(date_order.value).startswith('y'):
                            timestamp_note.content.value+=' The file value begins with the year. Choose Automatic or its year-first order.'
                        timestamp_note.content.color=NAMED_COLOURS['Red']
                else:
                    preview_text.value=f'File value: {time}  →  {value}'
            else: preview_text.value='Select the timestamp column to see an example row.'
            if update: self.page.update()
        for control in (time_format,timestamp,date,date_order,clock,activity,measure,flag_meaning): control.on_select=sync
        async def apply(_):
            if self.busy: return
            try:
                if timestamp.value not in columns: raise DataError('Select the timestamp column.')
                if time_format.value=='split' and date.value not in columns: raise DataError('Select the separate date column.')
                if time_format.value=='split' and date.value==timestamp.value: raise DataError('Select different date and time columns.')
                events=measure.value=='events'
                if not events and activity.value not in columns: raise DataError('Select the activity column.')
                if not events and activity.value in {timestamp.value,date.value if time_format.value=='split' else NONE}:
                    raise DataError('The activity column must be different from the timestamp columns.')
                requested=float(bins.value)
                Settings(bin_minutes=requested,schedule_enabled=False).validate()
                duration=60. if events else float(interval.value)*(60 if interval_unit.value=='minutes' else 1)
                if not math.isfinite(duration) or duration<=0: raise DataError('Enter a positive interval for each row.')
                if not events and (requested*60+1e-7<duration or not math.isclose(requested*60/duration,round(requested*60/duration),abs_tol=1e-7,rel_tol=0)):
                    raise DataError('The plot bin must contain a whole number of row intervals. For 1-minute rows, use a plot bin of 1, 5, 10, 15, 30 or 60 minutes.')
                if start.visible and not str(start.value or '').strip(): raise DataError('Enter the recording start date and time.')
                if start.visible: parse_datetime(str(start.value))
                if events:
                    if not str(end.value or '').strip(): raise DataError('Enter the recording end date and time.')
                    if parse_datetime(str(end.value))<=parse_datetime(str(start.value)): raise DataError('Recording end must be after recording start.')
                    if not continuous.value: raise DataError('Confirm continuous logging so that empty event bins can be treated as zero.')
                new=ImportOptions(timestamp=timestamp.value,activity=NONE if events else activity.value,
                    subject=subject.value or NONE,group=NONE,duration=NONE,observed=NONE,episodes=NONE,
                    date_column=date.value if time_format.value=='split' else NONE,
                    timestamp_kind='calendar' if time_format.value=='split' else time_format.value,
                    data_mode='events' if events else ('samples' if raw_schema and time_format.value=='calendar' and timestamp.value==lookup['iso_time'] else 'binned'),
                    units=flag_meaning.value if measure.value=='flag' else ('minutes' if measure.value=='minutes' else 'seconds'),
                    aggregation='sum' if measure.value=='counts' else 'activity',value_unit='Activity counts',
                    fallback_seconds=duration,sample_seconds=duration,raw_bin_minutes=requested,
                    time_mode=clock.value if time_format.value in {'calendar','split'} else ('utc' if time_format.value.startswith('unix') else 'wall'),
                    decimal=decimal.value,day_first=date_order.value in {'dmy','ydm'},
                    date_order=date_order.value if time_format.value in {'calendar','split'} else '',
                    recording_start=str(start.value or '').strip() if start.visible else '',
                    recording_end=str(end.value or '').strip() if events else '',continuous_events=bool(continuous.value) if events else False)
            except (TypeError,ValueError,OverflowError) as exc:
                self.error(str(exc) if isinstance(exc,DataError) else 'Enter valid numbers for the row interval and plot bin size.')
                return
            import_button.disabled=True
            cancel_button.disabled=True
            body.disabled=True
            self.page.update()
            success=await self.import_payloads(payloads,new,display_bin_minutes=requested,close_import=True)
            if not success:
                import_button.disabled=False
                cancel_button.disabled=False
                body.disabled=False
                self.page.update()
        sync(update=False)
        body=ft.Column([
            self.note(f'{len(payloads)} file(s) · {payloads[0][0]}\nCSV, TSV or delimited TXT with column headings. Files imported together must use the same columns, units and row interval. By default, each filename identifies one individual.'),
            ft.TextButton('Show / hide first rows',on_click=lambda e:toggle(preview)),preview,
            self.heading('1. Timestamps'),time_format,ft.Row([timestamp,date],spacing=12),date_order,timestamp_note,raw_note,
            self.heading('2. Activity'),measure,activity,flag_meaning,interval_row,activity_note,
            start,end,continuous,
            self.heading('3. Plot bins'),bins,preview_text,
            ft.TextButton('Individual IDs and file options…',on_click=lambda e:toggle(optional_fields)),optional_fields],
            width=720,height=580,scroll=ft.ScrollMode.AUTO,spacing=12)
        import_button=ft.Button('Import files',on_click=apply)
        cancel_button=ft.TextButton('Cancel',on_click=self.close_dialog)
        self.page.show_dialog(ft.AlertDialog(title=self.text('Import activity data',21,True),content=body,
            modal=True,actions=[cancel_button,import_button]))

    async def import_payloads(self,payloads,opts,display_bin_minutes=None,close_import=False):
        if self.busy: return False
        try: current=max(1,round(float(self.controls['bin_minutes'].value)*60))
        except (TypeError,ValueError): current=60
        if display_bin_minutes is not None: current=int(round(display_bin_minutes*60))
        elif opts.data_mode!='binned': current=int(round(opts.raw_bin_minutes*60))
        self.set_busy(True,'Reading and validating recordings…')
        old=list(self.subjects); success=False
        def work():
            incoming=[]; notes=[]
            for name,raw in payloads:
                try: subs,messages=decode_source(name,raw,opts)
                except Exception as exc: raise DataError(f'Import stopped at {name}. No changes were applied.\n\n{exc}') from exc
                incoming.extend(subs); notes.extend(messages)
            merged,messages=merge_subjects(old,incoming)
            b=recommended_bin_seconds(merged,current)
            Settings(bin_minutes=b/60,schedule_enabled=False).validate()
            if b!=current: messages.append(f'Plot bin set to {b/60:g} minutes to fit the imported row intervals.')
            return merged,notes+messages,b
        try:
            merged,messages,b=await asyncio.to_thread(work)
            self.subjects=merged; self.import_options=opts; self.logs.extend(messages)
            for s in self.subjects: self.ensure_group_colour(s.group)
            self.controls['bin_minutes'].value=f'{b/60:g}'
            if opts.data_mode=='samples' or any('logger_occupancy' in src for sub in merged for src in sub.sources):
                self.cfg.minimum_coverage=90.0
            if display_bin_minutes is not None or opts.aggregation!='activity' or opts.data_mode=='events' or opts.units=='binary_flag':
                self.controls['metric'].value='auto'
            if close_import: self.page.pop_dialog()
            self.refresh_data(); self.mark_dirty(); success=True
        except Exception as exc: self.error(str(exc))
        finally: self.set_busy(False)
        if success: await self.render()
        return success

    def _viewport_size(self):
        w=getattr(self.page,'width',None) or (getattr(self.page.window,'width',None) if not self.page.web else None) or 1440
        h=getattr(self.page,'height',None) or (getattr(self.page.window,'height',None) if not self.page.web else None) or 940
        return max(320,float(w)-466),max(240,float(h)-315)

    def show_preview(self,raw,inches):
        if not raw.startswith(b'\x89PNG\r\n\x1a\n'):
            raise DataError('The plot renderer did not return a valid PNG preview.')
        self.preview_png=raw; self.preview_inches=inches; self.zoom_factor=1.
        # Flet 1.0: src accepts base64 strings. Do not use legacy src_base64.
        self.image=ft.Image(src=base64.b64encode(raw).decode('ascii'),width=800,height=500,
                            fit=ft.BoxFit.CONTAIN,gapless_playback=True,
                            error_content=ft.Container(content=self.text('The PNG could not be displayed by this Flet client. Check that flet and flet-desktop are both 1.0.0. The image export remains available.',14),
                                                       padding=20,width=600,height=180,bgcolor='#FFF0E5'))
        self.preview_scroll=ft.Row([ft.GestureDetector(content=self.image,on_secondary_tap=self.copy_plot_to_clipboard)],scroll=ft.ScrollMode.AUTO,spacing=0,
                                  vertical_alignment=ft.CrossAxisAlignment.START)
        self.preview_host.content=ft.Column([self.preview_scroll],scroll=ft.ScrollMode.AUTO,
                                            expand=True,spacing=0,horizontal_alignment=ft.CrossAxisAlignment.START)
        self.size_preview(); self.page.update()

    async def copy_plot_to_clipboard(self,_=None):
        if self.preview_png is None: return
        if sys.platform!='win32' or self.page.web:
            self.error('Copy image by right-click is available in the Windows desktop application.')
            return
        try:
            await asyncio.to_thread(self._copy_png_windows,self.preview_png)
            self.inform('Plot copied to clipboard · paste with Ctrl+V')
        except Exception as exc:
            self.error('Could not copy the plot to the Windows clipboard: '+str(exc))

    @staticmethod
    def _copy_png_windows(raw):
        import ctypes
        from PIL import Image
        image=Image.open(io.BytesIO(raw)).convert('RGB')
        out=io.BytesIO()
        image.save(out,format='BMP')
        dib=out.getvalue()[14:]
        user32=ctypes.windll.user32
        kernel32=ctypes.windll.kernel32
        kernel32.GlobalAlloc.argtypes=[ctypes.c_uint,ctypes.c_size_t]
        kernel32.GlobalAlloc.restype=ctypes.c_void_p
        kernel32.GlobalLock.argtypes=[ctypes.c_void_p]
        kernel32.GlobalLock.restype=ctypes.c_void_p
        kernel32.GlobalUnlock.argtypes=[ctypes.c_void_p]
        kernel32.GlobalFree.argtypes=[ctypes.c_void_p]
        user32.SetClipboardData.argtypes=[ctypes.c_uint,ctypes.c_void_p]
        user32.SetClipboardData.restype=ctypes.c_void_p
        handle=kernel32.GlobalAlloc(0x0002,len(dib))
        if not handle: raise OSError('Clipboard allocation failed')
        ptr=kernel32.GlobalLock(handle)
        if not ptr:
            kernel32.GlobalFree(handle)
            raise OSError('Clipboard memory lock failed')
        ctypes.memmove(ptr,dib,len(dib))
        kernel32.GlobalUnlock(handle)
        if not user32.OpenClipboard(None):
            kernel32.GlobalFree(handle)
            raise OSError('Clipboard is in use')
        try:
            if not user32.EmptyClipboard(): raise OSError('Cannot clear clipboard')
            if not user32.SetClipboardData(8,handle): raise OSError('Cannot set clipboard image')
            handle=None
        finally:
            user32.CloseClipboard()
            if handle: kernel32.GlobalFree(handle)

    def size_preview(self):
        if self.image is None: return
        w,h=self._viewport_size(); iw,ih=self.preview_inches
        fit=min((w-20)/iw,(h-20)/ih)
        self.image.width=max(1,round(iw*fit*self.zoom_factor))
        self.image.height=max(1,round(ih*fit*self.zoom_factor))
        self.preview_scroll.width=w-12
        self.zoom_label.value='Fit' if abs(self.zoom_factor-1)<1e-6 else f'{self.zoom_factor*100:.0f}% of fit'

    def on_resize(self,_=None):
        if self.image is not None: self.size_preview(); self.page.update()

    def zoom_in(self,_=None):
        self.zoom_factor=min(8.,self.zoom_factor*1.25); self.size_preview(); self.page.update()

    def zoom_out(self,_=None):
        self.zoom_factor=max(.25,self.zoom_factor/1.25); self.size_preview(); self.page.update()

    def reset_view(self,_=None):
        if self.preview_png is not None:
            # Recreate scroll containers too, so an old scroll position cannot hide the fit image.
            self.show_preview(self.preview_png,self.preview_inches)

    def set_busy(self,value,message=''):
        self.busy=value; self.header_actions.disabled=value; self.left_body.disabled=value
        self.render_button.disabled=value; self.target.disabled=value; self.exports.disabled=value
        self.progress.visible=value
        if message: self.status.value=message; self.status.color=self.MUTED
        self.page.update()

    async def target_changed(self,_=None):
        if self.busy: return
        if self.analysis is not None and not self.dirty and self.target.value in self.analysis.panels:
            await self.refresh_preview()
        elif self.subjects:
            await self.render()

    async def render(self,_=None):
        if self.busy: return
        try:
            cfg=self.read_settings(); snapshot=[replace(s,sources=list(s.sources)) for s in self.subjects]
        except Exception as exc: self.error(str(exc)); return
        self.set_busy(True,'Preparing actogram…')
        try:
            analysis=await asyncio.to_thread(build_analysis,snapshot,cfg,dict(self.group_colours))
            key=self.target.value if self.target.value in analysis.panels else next(iter(analysis.panels))
            data,size=await asyncio.to_thread(render_bytes,analysis,key)
            self.analysis=analysis; self.cfg=cfg; self.current_key=key; self.target.value=key
            self.view_kind='actogram'; self.stats_snapshot=None; self.dirty=False
            self.show_preview(data,size); self.update_plot_summary(); self.logs.extend(analysis.notes)
        except Exception as exc:
            LOG.exception('Plot failed'); self.error(str(exc)); self.dirty=True
            self.status.value='Plot not updated · check the error and settings'
        finally: self.set_busy(False)

    async def refresh_preview(self):
        if self.busy or self.analysis is None: return
        self.set_busy(True,'Rendering selected individual / group…')
        try:
            key=self.target.value
            data,size=await asyncio.to_thread(render_bytes,self.analysis,key)
            self.current_key=key; self.view_kind='actogram'; self.stats_snapshot=None
            self.show_preview(data,size); self.update_plot_summary()
        except Exception as exc: self.error(str(exc))
        finally: self.set_busy(False)

    def update_plot_summary(self):
        a=self.analysis; p=a.panels[self.current_key]; cfg=a.settings
        valid=np.isfinite(p.values[a.row_indices]); active=p.binary[a.row_indices]==1
        self.status.value=f'{p.name} · {STYLES[cfg.style]} · {a.shown_days} days · {cfg.bin_minutes:g}-min bins'
        self.status.color=self.TEXT
        self.summary.value=f'{int(valid.sum()):,} bins with plotted values · {int(active.sum()):,} bins meet the activity threshold'
        if p.is_group: self.summary.value+=f' · {p.members} included individuals'
        if cfg.schedule_mode=='multiple' and cfg.schedule_enabled: self.summary.value+=f' · {len(cfg.treatments)} treatment periods'
        if not valid.any(): self.summary.value+=' · No values for the selected measure; missing is not zero.'
        self.page.update()

    def remove_included(self,_=None):
        def remove():
            self.subjects=[s for s in self.subjects if not s.included]
            self.analysis=None; self.image=None; self.preview_png=None
            self.preview_host.content=self.empty_preview(); self.refresh_data(); self.mark_dirty()
        self.confirm('Remove the included individuals from the workspace?',remove)


    def set_included(self, sid: str, include: bool):
        next(s for s in self.subjects if s.id == sid).included = bool(include)
        self.refresh_data(); self.mark_dirty()


    def include_all(self, _=None):
        for s in self.subjects:
            s.included = True
        self.refresh_data(); self.mark_dirty()


    def exclude_all(self, _=None):
        for s in self.subjects:
            s.included = False
        self.refresh_data(); self.mark_dirty()


    def assign_group(self, _=None):
        group = str(self.group_name.value or "").strip()
        if not group:
            self.error("Enter a group name first."); return
        if not any(s.included for s in self.subjects):
            self.error("Include at least one subject to assign."); return
        self.ensure_group_colour(group)
        for s in self.subjects:
            if s.included:
                s.group = group
        self.refresh_data(); self.mark_dirty()


    def ensure_group_colour(self, group: str):
        if group not in self.group_colours:
            palette = [NAMED_COLOURS[k] for k in ("Orange", "Teal", "Blue", "Purple", "Green", "Rose", "Brown")]
            self.group_colours[group] = palette[(len(self.group_colours) - 1) % len(palette)]


    def edit_subject(self, sid: str):
        s = next(s for s in self.subjects if s.id == sid)
        label = ft.TextField(label="Display name", value=s.name)
        group = ft.TextField(label="Group", value=s.group)
        inherit = ft.Checkbox(label="Use the group colour", value=s.colour is None)
        def apply(_):
            new_group = str(group.value or "").strip() or "Ungrouped"
            s.label = str(label.value or "").strip() or s.id
            s.group = new_group
            if inherit.value:
                s.colour = None
            self.ensure_group_colour(new_group)
            self.page.pop_dialog(); self.refresh_data(); self.mark_dirty()
        self.page.show_dialog(ft.AlertDialog(title=self.text(f"Subject · {sid}", 19, True),
            content=ft.Column([label, group, inherit], width=420, tight=True),
            actions=[ft.TextButton("Cancel", on_click=self.close_dialog), ft.Button("Apply", on_click=apply)]))


    def subject_colour_dialog(self, sid: str):
        s = next(s for s in self.subjects if s.id == sid)
        def apply(colour):
            s.colour = colour; self.refresh_data(); self.mark_dirty()
        self.colour_dialog(f"Subject colour · {s.name}", s.colour or self.group_colours.get(s.group, NAMED_COLOURS["Ink"]), apply)


    def set_group_colour(self, group: str, colour: str):
        self.group_colours[group] = colour
        self.refresh_data(); self.mark_dirty()


    def colour_dialog(self, title: str, initial: str, callback: Callable[[str], None]):
        h, s, v = colorsys.rgb_to_hsv(*to_rgb(initial))
        preview = ft.Container(height=65, bgcolor=initial, border_radius=12, border=ft.Border.all(1, self.BORDER))
        def recolour(_=None):
            preview.bgcolor = to_hex(colorsys.hsv_to_rgb(hue.value / 360, saturation.value / 100, brightness.value / 100))
            self.page.update()
        hue = ft.Slider(min=0, max=360, divisions=360, value=h * 360, on_change=recolour)
        saturation = ft.Slider(min=0, max=100, divisions=100, value=s * 100, on_change=recolour)
        brightness = ft.Slider(min=0, max=100, divisions=100, value=v * 100, on_change=recolour)
        def preset(colour: str):
            hh, ss, vv = colorsys.rgb_to_hsv(*to_rgb(colour))
            hue.value, saturation.value, brightness.value = hh * 360, ss * 100, vv * 100
            recolour()
        swatches = [ft.Container(width=37, height=37, bgcolor=colour, border_radius=9,
                    tooltip=name, border=ft.Border.all(1, self.BORDER),
                    on_click=lambda e, colour=colour: preset(colour)) for name, colour in NAMED_COLOURS.items()]
        def apply(_):
            chosen = preview.bgcolor
            self.page.pop_dialog()
            callback(chosen)
            self.page.update()
        self.page.show_dialog(ft.AlertDialog(title=self.text(title, 19, True),
            content=ft.Column([preview, self.text("Choose a named swatch or adjust the sliders.", 12, colour=self.MUTED),
                ft.Row(swatches, wrap=True, spacing=9, run_spacing=9),
                self.text("Hue"), hue, self.text("Saturation"), saturation, self.text("Brightness"), brightness],
                width=460, tight=True, scroll=ft.ScrollMode.AUTO),
            actions=[ft.TextButton("Cancel", on_click=self.close_dialog), ft.Button("Use this colour", on_click=apply)]))


    def palette_gallery(self, _=None):
        from matplotlib.image import imsave
        cards = []
        for key, name in CMAPS.items():
            if key in {"custom", "group"}:
                continue
            strip = colormaps[key](np.linspace(0, 1, 360))
            out = io.BytesIO()
            with MPL_LOCK:
                imsave(out, np.tile(strip[None, :, :], (20, 1, 1)), format="png")
            def choose(_, name=key):
                self.controls["cmap"].value = name
                self.page.pop_dialog(); self.mark_dirty()
            cards.append(ft.Container(content=ft.Column([self.text(name, 12, True), ft.Image(src=base64.b64encode(out.getvalue()).decode("ascii"), width=410, height=22)], spacing=5),
                                      padding=8, on_click=choose, border_radius=8, border=ft.Border.all(1, self.BORDER)))
        self.page.show_dialog(ft.AlertDialog(title=self.text("Choose a heatmap palette", 19, True),
            content=ft.Column(cards, width=450, height=500, scroll=ft.ScrollMode.AUTO),
            actions=[ft.TextButton("Close", on_click=self.close_dialog)]))


    def close_dialog(self, _=None):
        self.page.pop_dialog(); self.page.update()


    def error(self, message: str):
        LOG.error(message)
        self.logs.append("ERROR: " + message)
        self.page.show_dialog(ft.AlertDialog(title=self.text("Please check", 20, True),
            content=ft.Container(content=ft.Text(str(message), selectable=True, size=14), width=640),
            actions=[ft.TextButton("Close", on_click=self.close_dialog)]))
        self.page.update()


    def inform(self, message: str):
        self.logs.append(message)
        self.status.value = message
        self.status.color = self.MUTED
        self.page.update()


    async def pick_payloads(self, extensions: list[str], multiple: bool = False) -> list[tuple[str, bytes]]:
        picked = await ft.FilePicker().pick_files(allow_multiple=multiple,
            file_type=ft.FilePickerFileType.CUSTOM, allowed_extensions=extensions, with_data=bool(self.page.web))
        out = []
        total = 0
        for p in picked or []:
            raw = p.bytes if self.page.web else None
            if raw is None and p.path:
                path = Path(p.path)
                if path.stat().st_size > MAX_IMPORT_BYTES:
                    raise DataError(f"{p.name} exceeds 256 MB.")
                raw = await asyncio.to_thread(path.read_bytes)
            if raw is None:
                raise DataError("The selected file could not be read. Launch the desktop app with python actogram_Lab.py.")
            total += len(raw)
            if total > MAX_IMPORT_BYTES:
                raise DataError("The selected batch exceeds the 256 MB import limit. Import fewer files at a time.")
            out.append((p.name, raw))
        return out


    async def pick_csvs(self, _=None):
        if self.busy:
            return
        try:
            payloads = await self.pick_payloads(["csv", "tsv", "txt"], True)
            if payloads:
                await self.route_import(payloads)
        except Exception as exc:
            self.error(str(exc))


    async def pick_folder(self, _=None):
        if self.busy:
            return
        if self.page.web:
            self.error("Folder import is available in the desktop app. Use Add files in browser mode.")
            return
        try:
            folder = await ft.FilePicker().get_directory_path(dialog_title="Folder of activity data")
            if not folder:
                return
            paths = sorted(p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in {'.csv','.tsv','.txt'})
            if not paths:
                raise DataError("No CSV, TSV or delimited TXT files found in this folder.")
            if sum(p.stat().st_size for p in paths) > MAX_IMPORT_BYTES:
                raise DataError("This folder exceeds the import limit. Select a smaller batch using Add files.")
            payloads = await asyncio.to_thread(lambda: [(p.name, p.read_bytes()) for p in paths])
            await self.route_import(payloads)
        except Exception as exc:
            self.error(str(exc))


    def use_source_bin(self, _=None):
        if not any(s.included for s in self.subjects):
            self.error("Import and include at least one recording first.")
            return
        b = recommended_bin_seconds(self.subjects, 1)
        self.controls["bin_minutes"].value = f"{b / 60:g}"
        self.mark_dirty()


    def confirm(self, message: str, callback: Callable):
        async def yes(_):
            self.page.pop_dialog()
            result = callback()
            if asyncio.iscoroutine(result):
                await result
            self.page.update()
        self.page.show_dialog(ft.AlertDialog(title=self.text("Confirm workspace change", 19, True),
            content=ft.Text(message), actions=[ft.TextButton("Cancel", on_click=self.close_dialog), ft.Button("Continue", on_click=yes)]))


    async def save_bytes(self, raw: bytes, filename: str, extension: str):
        path = await ft.FilePicker().save_file(file_name=filename,
            file_type=ft.FilePickerFileType.CUSTOM, allowed_extensions=[extension], src_bytes=raw)
        if self.page.web:
            self.inform("The browser export request was sent.")
        elif path:
            self.inform(f"Saved: {path}")
        else:
            self.inform("Save cancelled.")


    def show_log(self, _=None):
        text = "\n\n".join(self.logs[-150:]) or "No import or analysis messages yet."
        self.page.show_dialog(ft.AlertDialog(title=self.text("Import & analysis log", 20, True),
            content=ft.Column([ft.Text(text, selectable=True, size=12)], width=800, height=520, scroll=ft.ScrollMode.AUTO),
            actions=[ft.TextButton("Close", on_click=self.close_dialog)]))

    def require_current(self):
        if self.analysis is None or self.dirty or self.current_key not in self.analysis.panels:
            raise DataError('Update the plot before exporting. Exports use the settings of the displayed plot.')

    async def export_image(self,fmt):
        if self.busy: return
        try:
            self.require_current()
            self.set_busy(True,f'Preparing {fmt.upper()} export…')
            if self.view_kind=='stats' and self.stats_snapshot is not None:
                table,cfg,a,b,measure=self.stats_snapshot
                raw,_=await asyncio.to_thread(render_comparison,table,cfg,a,b,measure,fmt,cfg.export_dpi)
                name=f'Activity_comparison.{fmt}'
            else:
                a=self.analysis; key=self.current_key
                raw,_=await asyncio.to_thread(render_bytes,a,key,fmt,a.settings.export_dpi)
                name=f'{safe_filename(a.panels[key].name)}_{a.settings.style}.{fmt}'
            await self.save_bytes(raw,name,fmt)
        except Exception as exc: self.error(str(exc))
        finally: self.set_busy(False)

    async def export_png(self,_=None): await self.export_image('png')
    async def export_svg(self,_=None): await self.export_image('svg')
    async def export_pdf(self,_=None): await self.export_image('pdf')

    async def export_csv(self,_=None):
        if self.busy: return
        try:
            self.require_current(); self.set_busy(True,'Preparing table…')
            if self.view_kind=='stats' and self.stats_snapshot is not None:
                raw=csv_bytes(self.stats_snapshot[0]); name='Individual_summaries.csv'
            else:
                a=self.analysis; key=self.current_key
                raw=await asyncio.to_thread(lambda:csv_bytes(panel_table(a,key)))
                name=f'{safe_filename(a.panels[key].name)}_bins.csv'
            await self.save_bytes(raw,name,'csv')
        except Exception as exc: self.error(str(exc))
        finally: self.set_busy(False)

    def batch_dialog(self,_=None):
        try: self.require_current()
        except Exception as exc: self.error(str(exc)); return
        kind=self.dropdown('Contents',{'all':'Individuals and groups','subjects':'Individuals only','groups':'Groups only'},'all')
        fmt=self.dropdown('Image format',{'png':'PNG','svg':'SVG','pdf':'PDF'},'png')
        async def export(_):
            self.page.pop_dialog(); self.set_busy(True,'Creating plots and tables…')
            try:
                raw=await asyncio.to_thread(batch_zip,self.analysis,kind.value,fmt.value)
                await self.save_bytes(raw,'Actogram_Lab_exports.zip','zip')
            except Exception as exc: self.error(str(exc))
            finally: self.set_busy(False)
        self.page.show_dialog(ft.AlertDialog(title=self.text('Batch export',20,True),
            content=ft.Column([kind,fmt,self.note('One plot and one single-plotted bin table per target, plus treatment summaries and the exact settings.')],width=470,tight=True),
            actions=[ft.TextButton('Cancel',on_click=self.close_dialog),ft.Button('Create ZIP',on_click=export)]))

    async def save_project(self,_=None):
        if self.busy: return
        try:
            cfg=self.read_settings(); self.set_busy(True,'Preparing project…')
            raw=await asyncio.to_thread(save_project_bytes,self.subjects,cfg,dict(self.group_colours),self.import_options)
            await self.save_bytes(raw,'Actogram_Lab.actoproject','actoproject')
        except Exception as exc: self.error(str(exc))
        finally: self.set_busy(False)

    async def open_project(self,_=None):
        if self.busy: return
        try:
            payloads=await self.pick_payloads(['actoproject'])
            if not payloads: return
            loaded=await asyncio.to_thread(load_project_bytes,payloads[0][1])
            async def apply():
                self.subjects,self.cfg,self.group_colours,self.import_options=loaded
                self.analysis=None; self.dirty=True; self.stats_snapshot=None
                self.load_settings_controls(); self.refresh_data(); self.navigate('data')
                self.logs.append(f'Opened {payloads[0][0]}.')
                if any(s.included for s in self.subjects): await self.render()
                else:
                    self.image=None; self.preview_png=None
                    self.preview_host.content=self.empty_preview(); self.page.update()
            if self.subjects: self.confirm('Replace the current workspace with this project? Unsaved changes will be lost.',apply)
            else: await apply()
        except Exception as exc: self.error(str(exc))

    async def summary_dialog(self,_=None):
        if self.busy: return
        try:
            cfg=self.read_settings(); snapshot=[replace(s,sources=list(s.sources)) for s in self.subjects]
            self.set_busy(True,'Summarising individuals and treatment periods…')
            analysis=await asyncio.to_thread(build_analysis,snapshot,cfg,dict(self.group_colours))
            table=await asyncio.to_thread(activity_statistics,analysis)
            if table.empty: raise DataError('No observed bins remain for a summary.')
        except Exception as exc:
            self.error(str(exc)); return
        finally: self.set_busy(False)
        conditions=list(dict.fromkeys(table.condition))
        choices={s:s for s in conditions}
        left=self.dropdown('Condition A',choices,conditions[0])
        right=self.dropdown('Condition B',choices,conditions[min(1,len(conditions)-1)])
        measures={'active_percent':'Time active / observed time (%)','total_activity_seconds':'Total activity (hours)',
                  'mean_plotted_intensity':'Mean displayed intensity'}
        if not table.total_activity_seconds.notna().any():
            measures={'mean_plotted_intensity':'Mean displayed intensity'}
        measure=self.dropdown('Comparison measure',measures,next(iter(measures)))
        async def compare(_):
            if left.value==right.value:
                self.error('Choose two different conditions.'); return
            self.page.pop_dialog(); self.set_busy(True,'Preparing comparison…')
            try:
                a,b,m=left.value,right.value,measure.value
                raw,size=await asyncio.to_thread(render_comparison,table,cfg,a,b,m)
                self.analysis=analysis; self.cfg=cfg; self.dirty=False
                self.current_key=self.target.value if self.target.value in analysis.panels else next(iter(analysis.panels))
                self.target.value=self.current_key; self.view_kind='stats'
                self.stats_snapshot=(table,cfg,a,b,m); self.show_preview(raw,size)
                self.status.value='Activity comparison · descriptive only'
                self.summary.value=f'{table.id.nunique()} individuals · {len(conditions)} recorded phase / treatment conditions'
            except Exception as exc: self.error(str(exc))
            finally: self.set_busy(False)
        async def export(_):
            self.page.pop_dialog(); self.set_busy(True,'Preparing summary tables…')
            try:
                def pack():
                    stream=io.BytesIO()
                    with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as z:
                        z.writestr('individual_summaries.csv',csv_bytes(table))
                        z.writestr('group_summaries.csv',csv_bytes(group_statistics(table)))
                        z.writestr('settings.json',json.dumps(asdict(cfg),ensure_ascii=False,indent=2))
                        z.writestr('INTERPRETATION.txt','\n'.join(analysis.notes)+'\nTotals depend on observation time. Boundary-crossing display bins are a separate condition; no fractional timing is inferred.\n')
                    return stream.getvalue()
                raw=await asyncio.to_thread(pack)
                await self.save_bytes(raw,'Actogram_Lab_summaries.zip','zip')
            except Exception as exc: self.error(str(exc))
            finally: self.set_busy(False)
        self.page.show_dialog(ft.AlertDialog(title=self.text('Activity summaries',20,True),
            content=ft.Column([self.note('Summaries follow the current date/time range and dated treatments.'),
                left,right,measure,self.note('Boundary interval means that a display bin crosses a phase or treatment transition. ')],width=660,tight=True,scroll=ft.ScrollMode.AUTO),
            actions=[ft.TextButton('Close',on_click=self.close_dialog),ft.OutlinedButton('Export tables',on_click=export),
                     ft.Button('Plot comparison',on_click=compare,disabled=len(conditions)<2)]))

    def show_help(self,_=None):
        message='''ACTOGRAM Lab

1. IMPORT DATA
Use Add files or Load folder. Native motion files containing bin_start_iso and motion_seconds load directly. Individual IDs come from filenames.

Use Convert TXT to CSV to preview tab-separated, space-separated, semicolon-separated, fixed-width or other text tables. Choose the separator and whether the first row contains headings. 


Other CSV, TSV or delimited TXT files open a short import form. They need column headings and a timestamp, plus an activity column unless each row is an event.  For other incompatible formats, prepare a matching table before importing.

Choose a combined date/time column, separate date and time columns, elapsed seconds/minutes with a recording start, or Unix seconds/milliseconds. Unix timestamps use UTC. 

Activity can be seconds active, minutes active, 0/1 or inactive/active flags, counts, or one event per row. Enter how much time each regular row covers: for example 1 second or 1 minute. Then choose the plot bin, such as 1, 10 or 30 minutes. A plot bin must contain a whole number of row intervals.

For flags, choose whether 1 means activity occurred at some point (count active rows), or active for the full interval (calculate active duration). 



Event-only logs need recording start/end and confirmation that logging was continuous. Empty bins inside that period become zero events; outside it, data remain missing. For regular rows, missing timestamps or blank measurements remain unknown. Enter 0 for an observed interval with no activity ( This feature isstill under construction - Hamid).

Individual IDs and file options lets you select an ID column when several individuals share one file, change the decimal separator, or convert time-zone offsets to UTC. Otherwise each filename identifies one individual. Set groups and display names in Data after importing.

2. PLOT
The ordinary view uses clock time and calendar dates. Choose days down or days across in Appearance ( Under construction ).  At very compact spacing, automatic date labels become less frequent to remain readable.

Start is included, end is excluded, and blank means the available range. Trim / exclude can omit the first or last hours and selected dates. 

3. LIGHT / DARK
Set lights on/off, choose constant light or darkness, or disable phase shading. Advanced treatments defines dated periods with individual cycles. Equal periods starts each new period exactly the chosen number of days after the previous one. Periods may touch but cannot overlap.

In Appearance, select light, dark or both phases and adjust Phase shading opacity,

4. SAVE AND EXPORT
Right click on the plot to copy and paste quickly. Use + / - and Fit to inspect the preview. Export PNG, SVG, PDF or the plotted table.  Summaries compares the observed activity between phases and treatments. Save project preserves recordings, import choices, treatments, groups and appearance.

Plots use fixed 24-hour days. When converting timestamps to UTC, enter the light schedule and plot limits in UTC too.'''
        self.page.show_dialog(ft.AlertDialog(title=self.text('Using Actogram Lab',20,True),
            content=ft.Column([ft.Text(message,selectable=True,size=13)],width=760,height=570,scroll=ft.ScrollMode.AUTO),
            actions=[ft.TextButton('Close',on_click=self.close_dialog)]))


def cli():
    parser=argparse.ArgumentParser(description='Actogram Lab: activity plots with optional dated illumination treatments.')
    parser.add_argument('--files',nargs='+',type=Path,help='Native CSV, TSV or delimited TXT files to preload.')
    parser.add_argument('--export-dir',type=Path,help='Export native files headlessly, without Flet.')
    parser.add_argument('--settings',type=Path,help='Optional JSON settings exported by this version.')
    parser.add_argument('--style',choices=list(STYLES),default=None)
    parser.add_argument('--bin-minutes',type=float,default=None)
    parser.add_argument('--start',default=None,help='Plot start, e.g. "2026-09-22 06:00".')
    parser.add_argument('--end',default=None,help='Exclusive plot end, e.g. "2026-09-29 06:00".')
    parser.add_argument('--web',action='store_true',help='Launch the browser UI instead of the desktop client.')
    parser.add_argument('--port',type=int,default=8550)
    args=parser.parse_args()
    try:
        cfg=Settings()
        if args.settings:
            settings=json.loads(args.settings.read_text(encoding='utf-8'))
            cfg=Settings(**{k:v for k,v in settings.items() if k in {f.name for f in fields(Settings)}})
        for arg,key in [('style','style'),('bin_minutes','bin_minutes'),('start','window_start'),('end','window_end')]:
            if getattr(args,arg) is not None: setattr(cfg,key,getattr(args,arg))
        cfg.validate()
        payloads=[]
        for p in args.files or []:
            if p.stat().st_size>MAX_IMPORT_BYTES: raise DataError(f'{p.name} exceeds the import limit.')
            payloads.append((p.name,p.read_bytes()))
        if sum(len(b) for _,b in payloads)>MAX_IMPORT_BYTES: raise DataError('Selected files exceed the 256 MB batch limit.')
        if args.export_dir:
            if not payloads: parser.error('--export-dir requires --files.')
            subjects=[]
            for name,raw in payloads:
                incoming,notes=decode_source(name,raw,ImportOptions(subject=NONE))
                subjects,more=merge_subjects(subjects,incoming)
                for note in notes+more: print(note)
            if args.bin_minutes is None:
                cfg.bin_minutes=recommended_bin_seconds(subjects,cfg.bin_seconds)/60
            analysis=build_analysis(subjects,cfg,{})
            args.export_dir.mkdir(parents=True,exist_ok=True)
            for key,p in analysis.panels.items():
                raw,_=render_bytes(analysis,key,dpi=cfg.export_dpi)
                stem=safe_filename(key)
                (args.export_dir/f'{stem}.png').write_bytes(raw)
                (args.export_dir/f'{stem}.csv').write_bytes(csv_bytes(panel_table(analysis,key)))
            (args.export_dir/'settings.json').write_text(json.dumps(asdict(cfg),indent=2),encoding='utf-8')
            (args.export_dir/'individual_summaries.csv').write_bytes(csv_bytes(activity_statistics(analysis)))
            print(f'Exported {len(analysis.panels)} plots to {args.export_dir}')
            return
        if ft is None:
            raise DataError('Flet is not installed. Run: python -m pip install "flet[desktop]==1.0.0"')
        from importlib.metadata import version
        installed=version('flet')
        if installed!=FLET_VERSION:
            raise DataError(f'This program targets Flet {FLET_VERSION}; this environment has {installed}.\nRun: python -m pip install --upgrade "flet[desktop]=={FLET_VERSION}"')
        async def start(page):
            app=ActogramApp(page); app.cfg=cfg; app.load_settings_controls(); app.navigate('data')
            if payloads: await app.route_import(payloads)
        if args.web: ft.run(start,view=ft.AppView.WEB_BROWSER,port=args.port)
        else: ft.run(start)
    except (DataError,OSError,ValueError) as exc:
        parser.exit(2,f'Actogram Lab: {exc}\n')

if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='%(levelname)s: %(message)s')
    cli()
