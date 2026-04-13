#!/usr/bin/perl
use strict;
use warnings;
use File::Path qw(make_path);
use File::Copy qw(copy);
use File::Spec;
use Getopt::Long qw(GetOptions);
use POSIX qw(strftime);

# ============================================================
# vLLM workable installation snapshot + comparison tool
#
# Default source install path:
#   /local_opt/vllm-install
#
# Default snapshot output path:
#   /local_opt/workable_llm_info
#
# Commands:
#   snapshot
#   compare --target /path/
#
# Examples:
#   perl vllm_workable_info.pl snapshot
#   perl vllm_workable_info.pl compare --target /path/

=b
perl vllm_workable_info.pl snapshot \
  --install-dir /local_opt/vllm-install \
  --out-dir /local_opt/workable_llm_info
  
perl vllm_workable_info.pl compare \
  --out-dir /local_opt/workable_llm_info \
  --target /path/


=cut
# ============================================================

my $DEFAULT_INSTALL = '/local_opt/vllm-install';
my $DEFAULT_OUTDIR  = '/local_opt/workable_llm_info';

my $command = shift @ARGV // '';

my %opt = (
    install_dir => $DEFAULT_INSTALL,
    out_dir     => $DEFAULT_OUTDIR,
    target      => '',
    help        => 0,
);

GetOptions(
    'install-dir=s' => \$opt{install_dir},
    'out-dir=s'     => \$opt{out_dir},
    'target=s'      => \$opt{target},
    'help|h'        => \$opt{help},
) or die usage();

if ($opt{help} || !$command) {
    print usage();
    exit($command ? 0 : 1);
}

if ($command eq 'snapshot') {
    do_snapshot(\%opt);
}
elsif ($command eq 'compare') {
    die "[FAIL] --target is required for compare\n" unless $opt{target};
    do_compare(\%opt);
}
else {
    die "[FAIL] Unknown command: $command\n" . usage();
}

exit 0;

# ------------------------------------------------------------
# Main functions
# ------------------------------------------------------------

sub do_snapshot {
    my ($opt) = @_;
    my $install_dir = $opt->{install_dir};
    my $out_dir     = $opt->{out_dir};

    check_dir($install_dir, "Install directory not found: $install_dir");
    check_file(venv_python($install_dir), "Virtualenv python not found");

    make_path($out_dir) unless -d $out_dir;

    my $meta_file      = File::Spec->catfile($out_dir, 'snapshot_meta.txt');
    my $freeze_file    = File::Spec->catfile($out_dir, 'requirements_freeze.txt');
    my $summary_file   = File::Spec->catfile($out_dir, 'env_summary.txt');
    my $help_file      = File::Spec->catfile($out_dir, 'api_server_help.txt');
    my $symbol_file    = File::Spec->catfile($out_dir, 'symbol_check.txt');
    my $cmake_copy     = File::Spec->catfile($out_dir, 'CMakeLists.final_working.txt');
    my $install_copy   = File::Spec->catfile($out_dir, 'install.final_working.sh');
    my $smoke_copy     = File::Spec->catfile($out_dir, 'smoke_test_vllm_model.final_working.sh');

    print "[INFO] Creating snapshot from: $install_dir\n";
    print "[INFO] Saving into: $out_dir\n";

    write_text($meta_file, build_meta_text($install_dir, $out_dir));

    run_cmd_to_file(
        build_env_cmd($install_dir, qq{python -m pip freeze}),
        $freeze_file,
    );

    run_cmd_to_file(
        build_env_cmd($install_dir, python_summary_snippet()),
        $summary_file,
    );

    run_cmd_to_file(
        build_env_cmd($install_dir, qq{python -m vllm.entrypoints.openai.api_server --help}),
        $help_file,
    );

    my $vllm_so = vllm_so_path($install_dir);
    if (-f $vllm_so) {
        run_cmd_to_file(
            qq{nm -D } . shell_quote($vllm_so) . qq{ | c++filt | grep -i cutlass_moe_mm_sm100 || true},
            $symbol_file,
        );
    } else {
        write_text($symbol_file, "[WARN] _C.abi3.so not found: $vllm_so\n");
    }

    my $cmake = File::Spec->catfile($install_dir, 'vllm', 'CMakeLists.txt');
    if (-f $cmake) {
        copy_or_warn($cmake, $cmake_copy);
    }

    my $cwd = Cwd::getcwd();
    my $install_sh = File::Spec->catfile($cwd, 'install.sh');
    my $smoke_sh   = File::Spec->catfile($cwd, 'smoke_test_vllm_model.sh');

    copy_or_warn($install_sh, $install_copy) if -f $install_sh;
    copy_or_warn($smoke_sh, $smoke_copy) if -f $smoke_sh;

    print "[PASS] Snapshot completed.\n";
    print "[INFO] Files written under: $out_dir\n";
}

sub do_compare {
    my ($opt) = @_;
    my $snapshot_dir = $opt->{out_dir};
    my $target_dir   = $opt->{target};

    check_dir($snapshot_dir, "Snapshot directory not found: $snapshot_dir");
    check_dir($target_dir, "Target installation directory not found: $target_dir");
    check_file(venv_python($target_dir), "Target virtualenv python not found");

    my $tmp_dir = File::Spec->catdir('/tmp', 'vllm_compare_' . timestamp());
    make_path($tmp_dir);

    my $target_freeze   = File::Spec->catfile($tmp_dir, 'target_requirements_freeze.txt');
    my $target_summary  = File::Spec->catfile($tmp_dir, 'target_env_summary.txt');
    my $target_help     = File::Spec->catfile($tmp_dir, 'target_api_server_help.txt');
    my $target_symbol   = File::Spec->catfile($tmp_dir, 'target_symbol_check.txt');

    print "[INFO] Comparing snapshot: $snapshot_dir\n";
    print "[INFO] Against target:    $target_dir\n";

    run_cmd_to_file(
        build_env_cmd($target_dir, qq{python -m pip freeze}),
        $target_freeze,
    );

    run_cmd_to_file(
        build_env_cmd($target_dir, python_summary_snippet()),
        $target_summary,
    );

    run_cmd_to_file(
        build_env_cmd($target_dir, qq{python -m vllm.entrypoints.openai.api_server --help}),
        $target_help,
    );

    my $target_so = vllm_so_path($target_dir);
    if (-f $target_so) {
        run_cmd_to_file(
            qq{nm -D } . shell_quote($target_so) . qq{ | c++filt | grep -i cutlass_moe_mm_sm100 || true},
            $target_symbol,
        );
    } else {
        write_text($target_symbol, "[WARN] _C.abi3.so not found: $target_so\n");
    }

    print "\n========== PACKAGE FREEZE DIFF ==========\n";
    run_cmd_allow_fail(
        qq{diff -u }
        . shell_quote(File::Spec->catfile($snapshot_dir, 'requirements_freeze.txt'))
        . qq{ }
        . shell_quote($target_freeze)
    );

    print "\n========== ENV SUMMARY DIFF ==========\n";
    run_cmd_allow_fail(
        qq{diff -u }
        . shell_quote(File::Spec->catfile($snapshot_dir, 'env_summary.txt'))
        . qq{ }
        . shell_quote($target_summary)
    );

    print "\n========== SYMBOL CHECK DIFF ==========\n";
    run_cmd_allow_fail(
        qq{diff -u }
        . shell_quote(File::Spec->catfile($snapshot_dir, 'symbol_check.txt'))
        . qq{ }
        . shell_quote($target_symbol)
    );

    print "\n========== QUICK STATUS ==========\n";
    quick_status($snapshot_dir, $target_dir, $target_summary, $target_symbol, $target_help);

    print "\n[PASS] Comparison finished.\n";
    print "[INFO] Temporary target files: $tmp_dir\n";
}

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

sub usage {
    return <<"USAGE";
Usage:
  perl vllm_workable_info.pl snapshot [--install-dir PATH] [--out-dir PATH]
  perl vllm_workable_info.pl compare --target PATH [--out-dir PATH]

Commands:
  snapshot
      Save complete information for the current workable vLLM installation.

  compare
      Compare a new installation against the saved workable snapshot.

Options:
  --install-dir PATH
      Current workable installation path.
      Default: /local_opt/vllm-install

  --out-dir PATH
      Snapshot output path.
      Default: /local_opt/workable_llm_info

  --target PATH
      New installation path to compare against the snapshot.

Examples:
  perl vllm_workable_info.pl snapshot

  perl vllm_workable_info.pl compare --target /path/

USAGE
}

sub build_meta_text {
    my ($install_dir, $out_dir) = @_;
    my $time = strftime('%Y-%m-%d %H:%M:%S', localtime);
    return <<"TXT";
snapshot_time=$time
install_dir=$install_dir
out_dir=$out_dir
venv_python=@{[ venv_python($install_dir) ]}
vllm_so=@{[ vllm_so_path($install_dir) ]}
TXT
}

sub python_summary_snippet {
    return <<'PY';
python - <<'EOF'
import inspect
import json
import os

out = {}

try:
    import torch
    out["torch_version"] = torch.__version__
    out["cuda_available"] = bool(torch.cuda.is_available())
except Exception as e:
    out["torch_error"] = repr(e)

try:
    import triton
    out["triton_version"] = triton.__version__
    out["triton_path"] = inspect.getfile(triton)
except Exception as e:
    out["triton_error"] = repr(e)

try:
    import flashinfer
    out["flashinfer_path"] = inspect.getfile(flashinfer)
    out["flashinfer_version"] = getattr(flashinfer, "__version__", None)
except Exception as e:
    out["flashinfer_error"] = repr(e)

try:
    import vllm
    out["vllm_file"] = getattr(vllm, "__file__", None)
    out["vllm_path"] = list(getattr(vllm, "__path__", []))
    out["vllm_version"] = getattr(vllm, "__version__", None)
except Exception as e:
    out["vllm_error"] = repr(e)

try:
    import transformers
    out["transformers_version"] = transformers.__version__
except Exception as e:
    out["transformers_error"] = repr(e)

try:
    import tokenizers
    out["tokenizers_version"] = tokenizers.__version__
except Exception as e:
    out["tokenizers_error"] = repr(e)

try:
    import numpy
    out["numpy_version"] = numpy.__version__
except Exception as e:
    out["numpy_error"] = repr(e)

print(json.dumps(out, indent=2, sort_keys=True))
EOF
PY
}

sub quick_status {
    my ($snapshot_dir, $target_dir, $target_summary, $target_symbol, $target_help) = @_;

    my $snapshot_summary = File::Spec->catfile($snapshot_dir, 'env_summary.txt');
    my $snapshot_symbol  = File::Spec->catfile($snapshot_dir, 'symbol_check.txt');

    my $snap = slurp($snapshot_summary);
    my $tgt  = slurp($target_summary);
    my $sym  = slurp($target_symbol);
    my $help = slurp($target_help);

    print status_line("Target venv exists", -f venv_python($target_dir));
    print status_line("Target _C.abi3.so exists", -f vllm_so_path($target_dir));

    print status_line("Target triton is 3.5.0", $tgt =~ /"triton_version"\s*:\s*"3\.5\.0/);
    print status_line("Target transformers is 4.56.0", $tgt =~ /"transformers_version"\s*:\s*"4\.56\.0"/);
    print status_line("Target tokenizers is 0.22.2", $tgt =~ /"tokenizers_version"\s*:\s*"0\.22\.2"/);
    print status_line("Target numpy is 2.2.6", $tgt =~ /"numpy_version"\s*:\s*"2\.2\.6"/);
    print status_line("Target CUDA available", $tgt =~ /"cuda_available"\s*:\s*true/);
    print status_line("Target symbol defined", $sym =~ /\sT\s+cutlass_moe_mm_sm100/);
    print status_line("Target api_server help works", length($help) > 0 && $help !~ /Traceback/);
}

sub status_line {
    my ($name, $ok) = @_;
    return sprintf("[%s] %s\n", ($ok ? 'OK' : 'NO'), $name);
}

sub build_env_cmd {
    my ($install_dir, $inner_cmd) = @_;
    my $venv = shell_quote(venv_activate($install_dir));
    my $src  = shell_quote(File::Spec->catdir($install_dir, 'vllm'));

    return qq{bash -lc 'source $venv && cd $src && export PYTHONPATH=$src:\${PYTHONPATH:-} && $inner_cmd'};
}

sub venv_activate {
    my ($install_dir) = @_;
    return File::Spec->catfile($install_dir, '.vllm', 'bin', 'activate');
}

sub venv_python {
    my ($install_dir) = @_;
    return File::Spec->catfile($install_dir, '.vllm', 'bin', 'python');
}

sub vllm_so_path {
    my ($install_dir) = @_;
    return File::Spec->catfile($install_dir, 'vllm', 'vllm', '_C.abi3.so');
}

sub shell_quote {
    my ($s) = @_;
    $s =~ s/'/'"'"'/g;
    return "'$s'";
}

sub run_cmd_to_file {
    my ($cmd, $outfile) = @_;
    print "[INFO] Writing: $outfile\n";
    my $full = "$cmd > " . shell_quote($outfile) . " 2>&1";
    my $rc = system($full);
    if ($rc != 0) {
        my $exit = $rc >> 8;
        print "[WARN] Command exited with code $exit while writing $outfile\n";
    }
}

sub run_cmd_allow_fail {
    my ($cmd) = @_;
    print "\$ $cmd\n";
    system($cmd);
}

sub slurp {
    my ($path) = @_;
    return '' unless -f $path;
    open my $fh, '<', $path or return '';
    local $/ = undef;
    my $txt = <$fh>;
    close $fh;
    return defined $txt ? $txt : '';
}

sub write_text {
    my ($path, $text) = @_;
    open my $fh, '>', $path or die "[FAIL] Cannot write $path: $!\n";
    print {$fh} $text;
    close $fh;
}

sub copy_or_warn {
    my ($src, $dst) = @_;
    if (copy($src, $dst)) {
        print "[INFO] Copied $src -> $dst\n";
    } else {
        print "[WARN] Failed to copy $src -> $dst: $!\n";
    }
}

sub check_dir {
    my ($path, $msg) = @_;
    die "[FAIL] $msg\n" unless -d $path;
}

sub check_file {
    my ($path, $msg) = @_;
    die "[FAIL] $msg\n" unless -f $path;
}

sub timestamp {
    return strftime('%Y%m%d_%H%M%S', localtime);
}