#!/usr/bin/env python3
"""
Batch Renderer with Multiprocessing
Renders multiple model+HDR combinations in parallel using multiple Blender instances
"""
import argparse
import json
import subprocess
import sys
import textwrap
import time
import uuid
from datetime import datetime
from itertools import product
from multiprocessing import Pool, cpu_count
from pathlib import Path


def iso_now() -> str:
    """Return an ISO-8601 UTC timestamp without microseconds."""
    return datetime.utcnow().replace(microsecond=0).isoformat() + 'Z'


def build_job_key(model_path: Path, hdr_paths, resolution: int, pitch: int,
                  yaw_per_pitch: int, hdr_switch: int, output_format: str) -> str:
    """Create a stable identifier for a render job."""
    hdr_segment = '|'.join(str(h) for h in hdr_paths)
    parts = [
        str(model_path),
        hdr_segment,
        str(resolution),
        str(pitch),
        str(yaw_per_pitch),
        str(hdr_switch),
        output_format,
    ]
    return '::'.join(parts)


def save_progress(progress_path: Path, session_uuid: str, created_at: str,
                  total_jobs: int, job_states: dict) -> None:
    """Persist current progress to disk (atomic write)."""
    data = {
        'session_uuid': session_uuid,
        'created_at': created_at,
        'updated_at': iso_now(),
        'total_jobs': total_jobs,
        'jobs': job_states,
    }

    progress_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = progress_path.with_suffix('.tmp')
    with tmp_path.open('w', encoding='utf-8') as handle:
        json.dump(data, handle, indent=2)
    tmp_path.replace(progress_path)


def render_single(args):
    """Render a single model+HDR combination"""
    (job_key, model_path, hdr_paths, output_dir, resolution, pitch_steps, yaw_per_pitch,
     hdr_switch_interval, output_format, session_uuid, display_idx, display_total,
     use_augmented, global_idx, global_total) = args

    model_name = model_path.stem[:16]

    print(
        f"\n[{display_idx + 1}/{display_total}] 🎬 Rendering: {model_name} × {len(hdr_paths)} HDRs"
        f" (job {global_idx + 1}/{global_total})"
    )
    print(f"            Model: {model_path.name}")
    print(f"            HDRs:  {len(hdr_paths)} environments")

    start_time = time.time()

    if use_augmented:
        # Augmented renderer with high-res + JPEG XL
        hdr_paths_str = ','.join([str(h) for h in hdr_paths])
        cmd = [
            'blender',
            '--background',
            '--python', 'blender_renderer/render_dataset_augmented.py',
            '--',
            str(model_path),
            hdr_paths_str,
            str(output_dir),
            str(resolution),
            str(pitch_steps),
            str(yaw_per_pitch),
            str(hdr_switch_interval),
            output_format,
            session_uuid,
        ]
    else:
        # Legacy flat renderer
        cmd = [
            'blender',
            '--background',
            '--python', 'blender_renderer/render_dataset_flat.py',
            '--',
            str(model_path),
            str(hdr_paths[0]),
            str(output_dir),
            str(pitch_steps),
            str(yaw_per_pitch),
            session_uuid,
        ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout per render
        )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            print(
                f"[{display_idx + 1}/{display_total}] ✅ Complete: {model_name} × {len(hdr_paths)} HDRs"
                f" ({elapsed:.1f}s)"
            )
            return job_key, True, elapsed, ''
        else:
            print(f"[{display_idx + 1}/{display_total}] ❌ Failed: {model_name} × {len(hdr_paths)} HDRs")
            print(f"            Error: {result.stderr[-500:]}")  # Last 500 chars
            return job_key, False, elapsed, result.stderr[-500:]

    except subprocess.TimeoutExpired:
        print(f"[{display_idx + 1}/{display_total}] ⏱️  Timeout: {model_name} × {len(hdr_paths)} HDRs")
        return job_key, False, 600, 'timeout'
    except Exception as e:
        print(f"[{display_idx + 1}/{display_total}] ❌ Exception: {model_name} × {len(hdr_paths)} HDRs - {e}")
        return job_key, False, 0, str(e)


def main():
    parser = argparse.ArgumentParser(
        description='Batch render SOD datasets with multiprocessing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # High-res augmented rendering (2048px, JPEG XL, 40 views)
  python3 batch_render.py \\
      --models blenderkit-scraper/downloads/model/*.blend \\
      --hdrs blenderkit-scraper/downloads/hdr/*.exr \\
      --output datasets_highres \\
      --workers 4 \\
      --resolution 2048 \\
      --pitch 10 --yaw-per-pitch 4 \\
      --format jxl \\
      --augmented

  # Legacy flat rendering (640px, PNG, grid pattern)
  python3 batch_render.py \\
      --models models/*.blend \\
      --hdrs hdrs/*.exr \\
      --output datasets \\
      --workers auto \\
      --pitch 8 --yaw-per-pitch 12
        """
    )

    parser.add_argument('--models', nargs='+', required=True,
                       help='Model .blend files (supports wildcards)')
    parser.add_argument('--hdrs', nargs='+', required=True,
                       help='HDR .exr files (supports wildcards)')
    parser.add_argument('--output', type=str, default='datasets',
                       help='Output directory (default: datasets)')
    parser.add_argument('--workers', type=str, default='1',
                       help='Number of parallel workers (default: 1, use "auto" for CPU count)')
    parser.add_argument('--resolution', type=int, default=640,
                       help='Output resolution (default: 640, use 2048 or 4096 for high-res)')
    parser.add_argument('--pitch', type=int, default=10,
                       help='Number of pitch angles (default: 10)')
    parser.add_argument('--yaw-per-pitch', type=int, default=4,
                       help='Random yaw angles per pitch (default: 4)')
    parser.add_argument('--hdr-switch', type=int, default=4,
                       help='Switch HDR every N images (default: 4, 0=no switching)')
    parser.add_argument('--format', type=str, default='jxl', choices=['png', 'jxl'],
                       help='Output format (default: jxl)')
    parser.add_argument('--augmented', action='store_true',
                       help='Use augmented renderer (random angles, HDR rotation, etc.)')

    args = parser.parse_args()

    # Resolve paths
    model_paths = [Path(m).resolve() for m in args.models]
    hdr_paths = [Path(h).resolve() for h in args.hdrs]
    output_dir = Path(args.output).resolve()

    # Validate files exist
    model_paths = [m for m in model_paths if m.exists()]
    hdr_paths = [h for h in hdr_paths if h.exists()]

    if not model_paths:
        print("❌ No valid model files found")
        sys.exit(1)

    if not hdr_paths:
        print("❌ No valid HDR files found")
        sys.exit(1)

    # Determine worker count
    if args.workers == 'auto':
        workers = cpu_count()
    else:
        workers = int(args.workers)

    # Determine rendering mode
    use_augmented = args.augmented
    if args.resolution > 640 or args.format == 'jxl' or args.hdr_switch > 0:
        use_augmented = True  # Auto-enable augmented mode

    if use_augmented:
        combos = [(model, hdr_paths) for model in model_paths]
        mode_str = f"AUGMENTED ({args.resolution}px {args.format.upper()})"
    else:
        combos = [(model, [hdr]) for model, hdr in product(model_paths, hdr_paths)]
        mode_str = f"LEGACY ({args.resolution}px {args.format.upper()})"

    total_jobs = len(combos)
    views_per_job = args.pitch * args.yaw_per_pitch
    total_images = total_jobs * views_per_job

    progress_path = output_dir / 'batch_progress.json'
    job_states = {}
    created_at = iso_now()
    resuming = False

    if progress_path.exists():
        try:
            with progress_path.open('r', encoding='utf-8') as fh:
                stored_progress = json.load(fh)
            job_states = stored_progress.get('jobs', {}) or {}
            session_uuid = stored_progress.get('session_uuid') or str(uuid.uuid4())[:8]
            created_at = stored_progress.get('created_at', created_at)
            resuming = True
        except Exception as exc:
            print(f"⚠️  Failed to load progress file ({exc}). Starting new session.")
            session_uuid = str(uuid.uuid4())[:8]
            job_states = {}
    else:
        session_uuid = str(uuid.uuid4())[:8]

    all_jobs = []
    already_done = 0
    for global_idx, (model, hdr_list) in enumerate(combos):
        hdr_switch_val = args.hdr_switch if use_augmented else 0
        job_key = build_job_key(
            model,
            hdr_list,
            args.resolution,
            args.pitch,
            args.yaw_per_pitch,
            hdr_switch_val,
            args.format,
        )

        all_jobs.append(
            {
                'key': job_key,
                'model': model,
                'hdrs': hdr_list,
                'hdr_switch': hdr_switch_val,
                'global_idx': global_idx,
            }
        )

        state = job_states.get(job_key)
        if not state:
            job_states[job_key] = {
                'model': str(model),
                'hdrs': [str(h) for h in hdr_list],
                'resolution': args.resolution,
                'pitch': args.pitch,
                'yaw_per_pitch': args.yaw_per_pitch,
                'hdr_switch': hdr_switch_val,
                'format': args.format,
                'status': 'pending',
                'attempts': 0,
            }
            state = job_states[job_key]
        else:
            state.setdefault('status', 'pending')
            state.setdefault('attempts', 0)
            state.setdefault('model', str(model))
            state.setdefault('hdrs', [str(h) for h in hdr_list])
            state.setdefault('resolution', args.resolution)
            state.setdefault('pitch', args.pitch)
            state.setdefault('yaw_per_pitch', args.yaw_per_pitch)
            state.setdefault('hdr_switch', hdr_switch_val)
            state.setdefault('format', args.format)

        if state.get('status') == 'done':
            already_done += 1

    pending_jobs = []
    for job in all_jobs:
        state = job_states[job['key']]
        if state.get('status') == 'done':
            continue
        pending_jobs.append(job)

    remaining_jobs = len(pending_jobs)
    remaining_images = remaining_jobs * views_per_job

    requested_workers = workers
    if remaining_jobs > 0:
        workers = max(1, min(workers, remaining_jobs))

    workers_display = (
        f"{requested_workers} (using {workers})" if workers != requested_workers else str(workers)
    )

    table_width = 70
    inner_width = table_width

    def build_table_lines(label: str, value, align: str = 'left') -> list[str]:
        """Build wrapped table lines that stay within the ASCII border."""
        label_text = f"{label}: "
        available = inner_width - len(label_text)
        value_str = str(value)

        if available <= 0:
            wrapped = textwrap.wrap(f"{label}: {value_str}", width=inner_width) or ['']
            return [line.ljust(inner_width) for line in wrapped]

        if align == 'right' and len(value_str) <= available:
            line = label_text + value_str.rjust(available)
            return [line.ljust(inner_width)]

        wrapped = textwrap.wrap(value_str, width=available) or ['']
        lines = []
        first_line = label_text + wrapped[0]
        lines.append(first_line.ljust(inner_width))
        indent = ' ' * len(label_text)
        for segment in wrapped[1:]:
            lines.append((indent + segment).ljust(inner_width))
        return lines

    def print_table_lines(label: str, value, align: str = 'left') -> None:
        for line in build_table_lines(label, value, align):
            print(f"║{line}║")

    print(f"╔{'═'*table_width}╗")
    print(f"║{mode_str:^{table_width}}║")
    print(f"╠{'═'*table_width}╣")
    print_table_lines('Models', len(model_paths), align='right')
    print_table_lines('HDRs', len(hdr_paths), align='right')
    print_table_lines('Jobs', total_jobs, align='right')
    print_table_lines('Completed', already_done, align='right')
    print_table_lines('Pending', remaining_jobs, align='right')
    print_table_lines('Views per job', f"{args.pitch} pitch × {args.yaw_per_pitch} yaw = {views_per_job}")
    print_table_lines('Total images', total_images, align='right')
    print_table_lines('Images remaining', remaining_images, align='right')
    print_table_lines('Workers', workers_display, align='right')
    print_table_lines('Output', output_dir)
    print(f"╚{'═'*table_width}╝")

    if resuming:
        print(
            f"\n♻️  Resuming session {session_uuid} (created {created_at})."
            f" {already_done}/{total_jobs} jobs already completed."
        )
    else:
        print(f"\n🔖 Session UUID: {session_uuid}\n")

    if remaining_jobs == 0:
        save_progress(progress_path, session_uuid, created_at, total_jobs, job_states)
        print("✅ Nothing to render. All jobs are already complete.")
        print(f"\n📁 Output: {output_dir}")
        print(f"📝 Progress: {progress_path}")
        return

    if remaining_jobs > 10:
        response = input(f"\n⚠️  Render {remaining_images} images? [y/N]: ")
        if response.lower() != 'y':
            print("❌ Cancelled")
            sys.exit(0)

    # Prepare job arguments and mark as queued
    timestamp_now = iso_now()
    job_args = []
    for display_idx, job in enumerate(pending_jobs):
        state = job_states[job['key']]
        state['status'] = 'queued'
        state['updated_at'] = timestamp_now
        job_args.append(
            (
                job['key'],
                job['model'],
                job['hdrs'],
                output_dir,
                args.resolution,
                args.pitch,
                args.yaw_per_pitch,
                job['hdr_switch'],
                args.format,
                session_uuid,
                display_idx,
                remaining_jobs,
                use_augmented,
                job['global_idx'],
                total_jobs,
            )
        )

    save_progress(progress_path, session_uuid, created_at, total_jobs, job_states)

    # Render in parallel
    print(f"\n🚀 Starting {workers} parallel workers...\n")
    start_time = time.time()
    total_render_time = 0.0
    results = []

    def handle_result(result) -> None:
        nonlocal total_render_time
        job_key, success, job_elapsed, error_tail = result
        total_render_time += job_elapsed
        results.append(result)
        state = job_states.get(job_key, {})
        state['attempts'] = state.get('attempts', 0) + 1
        state['last_elapsed'] = job_elapsed
        timestamp = iso_now()
        state['updated_at'] = timestamp
        if success:
            state['status'] = 'done'
            state['completed_at'] = timestamp
            state.pop('error', None)
        else:
            state['status'] = 'failed'
            state['error'] = error_tail
        save_progress(progress_path, session_uuid, created_at, total_jobs, job_states)

    if workers == 1:
        for args_tuple in job_args:
            handle_result(render_single(args_tuple))
    else:
        with Pool(workers) as pool:
            for result in pool.imap_unordered(render_single, job_args):
                handle_result(result)

    elapsed = time.time() - start_time

    run_successful = sum(1 for _, success, _, _ in results if success)
    run_failed = sum(1 for _, success, _, _ in results if not success)
    overall_done = sum(1 for state in job_states.values() if state.get('status') == 'done')
    overall_failed = sum(1 for state in job_states.values() if state.get('status') == 'failed')
    overall_pending = total_jobs - overall_done - overall_failed

    save_progress(progress_path, session_uuid, created_at, total_jobs, job_states)

    print(f"\n╔{'═'*table_width}╗")
    print(f"║{'BATCH RENDERING COMPLETE':^{table_width}}║")
    print(f"╠{'═'*table_width}╣")
    print_table_lines('Successful (run)', run_successful, align='right')
    print_table_lines('Failed (run)', run_failed, align='right')
    print_table_lines('Overall done', f"{overall_done}/{total_jobs}", align='right')
    print_table_lines('Remaining jobs', overall_pending, align='right')
    print_table_lines('Wall time', f"{elapsed/60:.1f} min", align='right')
    print_table_lines('Total CPU time', f"{total_render_time/60:.1f} min", align='right')
    speedup = total_render_time/elapsed if elapsed > 0 else 0
    print_table_lines('Speedup', f"{speedup:.2f}x", align='right')
    avg_per_job = total_render_time / len(results) if results else 0.0
    print_table_lines('Avg per combo', f"{avg_per_job:.1f}s", align='right')
    print(f"╚{'═'*table_width}╝")

    print(f"\n📁 Output: {output_dir}")
    print(f"📝 Progress: {progress_path}")


if __name__ == '__main__':
    main()
