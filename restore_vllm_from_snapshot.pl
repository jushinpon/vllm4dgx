#!/usr/bin/perl
use strict;
use warnings;
use Getopt::Long qw(GetOptions);
use File::Spec;

my %opt = (
    install_dir => '/local_opt/vllm-install',
    info_dir    => '/local_opt/workable_llm_info',
    dry_run     => 0,
    help        => 0,
);

GetOptions(
    'install-dir=s' => \$opt{install_dir},
    'info-dir=s'    => \$opt{info_dir},
    'dry-run!'      => \$opt{dry_run},
    'help|h'        => \$opt{help},
) or die usage();

if ($opt{help}) {
    print usage();
    exit 0;
}

main();
exit 0;

sub main {
    my $install = $opt{install_dir};
    my $info    = $opt{info_dir};

    check_dir($install, "Install directory not found: $install");
    check_dir($info, "Snapshot info directory not found: $info");
    check_file(venv_python($install), "Virtualenv python not found: " . venv_python($install));

    print "\n========== vLLM Restore From Snapshot ==========\n";
    print "[INFO] install_dir = $install\n";
    print "[INFO] info_dir    = $info\n";
    print "[INFO] dry_run     = $opt{dry_run}\n";

    show_current_state($install);

    fix_python_packages($install);
    restore_cmake_if_needed($install, $info);
    restore_triton($install);
    rebuild_vllm($install);
    final_verify($install);

    print "\n[PASS] Restore procedure finished.\n";
}

sub fix_python_packages {
    my ($install) = @_;

    section("Fix Python package pins");

    my $cmd = env_cmd($install, q{
python -m pip install --force-reinstall \
  "numpy==2.2.6" \
  "transformers==4.56.0" \
  "tokenizers==0.22.2"
});

    run($cmd);
}

sub restore_cmake_if_needed {
    my ($install, $info) = @_;

    section("Check / restore CMake patch");

    my $cmake = File::Spec->catfile($install, 'vllm', 'CMakeLists.txt');
    check_file($cmake, "CMakeLists.txt not found: $cmake");

    my $text = slurp($cmake);

    if ($text =~ /target_sources\(_C PRIVATE\s+csrc\/quantization\/w8a8\/cutlass\/moe\/grouped_mm_c3x_sm100\.cu\s+\)/s) {
        print "[OK] target_sources(_C ... grouped_mm_c3x_sm100.cu) patch exists\n";
        return;
    }

    print "[WARN] CMake target_sources patch missing. Applying patch...\n";

    my $patch_cmd = qq{cd } . shell_quote(vllm_src_dir($install)) . q{ && python - <<'PY'
from pathlib import Path

p = Path("CMakeLists.txt")
text = p.read_text()

text = text.replace(
    'cuda_archs_loose_intersection(SCALED_MM_ARCHS "10.0f;11.0f" "${CUDA_ARCHS}")',
    'cuda_archs_loose_intersection(SCALED_MM_ARCHS "10.0f;11.0f;12.0f" "${CUDA_ARCHS}")'
)

text = text.replace(
    'cuda_archs_loose_intersection(SCALED_MM_ARCHS "10.0a" "${CUDA_ARCHS}")',
    'cuda_archs_loose_intersection(SCALED_MM_ARCHS "10.0a;10.1a;10.3a;12.0a;12.1a" "${CUDA_ARCHS}")'
)

anchor = 'target_compile_definitions(_C PRIVATE CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1)\n'
insert = anchor + '''
target_sources(_C PRIVATE
  csrc/quantization/w8a8/cutlass/moe/grouped_mm_c3x_sm100.cu
)
'''

if 'target_sources(_C PRIVATE\n  csrc/quantization/w8a8/cutlass/moe/grouped_mm_c3x_sm100.cu\n)' not in text:
    if anchor not in text:
        raise SystemExit("Cannot find target_compile_definitions anchor")
    text = text.replace(anchor, insert, 1)

p.write_text(text)
print("Patched CMakeLists.txt")
PY};

    run($patch_cmd);
}

sub restore_triton {
    my ($install) = @_;

    section("Restore pinned Triton 3.5.0");

    my $triton_dir = triton_src_dir($install);
    check_dir($triton_dir, "Triton source directory not found: $triton_dir");

    run(env_cmd($install, q{python -m pip uninstall -y triton >/dev/null 2>&1 || true}));

    my $cmd = env_cmd($install,
        q{cd } . shell_quote($triton_dir) . q{ && } .
        q{export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas && } .
        q{python -m pip install --no-build-isolation -v .}
    );

    run($cmd);

    run(env_cmd($install, q{python - <<'PY'
import triton, inspect
print("triton:", triton.__version__)
print("path:", inspect.getfile(triton))
assert triton.__version__.startswith("3.5.0")
PY}));
}

sub rebuild_vllm {
    my ($install) = @_;

    section("Rebuild vLLM");

    my $vllm_dir = vllm_src_dir($install);
    check_dir($vllm_dir, "vLLM source directory not found: $vllm_dir");

    my $cmd = env_cmd($install,
        q{cd } . shell_quote($vllm_dir) . q{ && } .
        q{rm -rf build dist .pytest_cache .setuptools-cmake-build vllm.egg-info && } .
        q{find . -name '*.so' -delete || true && } .
        q{find . -name '*.o' -delete || true && } .
        q{find . -name '__pycache__' -type d -prune -exec rm -rf {} + || true && } .
        q{export TORCH_CUDA_ARCH_LIST=12.1a && } .
        q{export VLLM_USE_FLASHINFER_MXFP4_MOE=1 && } .
        q{export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas && } .
        q{python -m pip install --no-build-isolation --no-deps -e .}
    );

    run($cmd);
}

sub final_verify {
    my ($install) = @_;

    section("Final verification");

    my $vllm_so = vllm_so_path($install);
    check_file($vllm_so, "_C.abi3.so not found: $vllm_so");

    run(env_cmd($install, q{python - <<'PY'
import torch, triton, transformers, tokenizers, numpy, flashinfer, vllm, inspect
print("torch:", torch.__version__)
print("triton:", triton.__version__)
print("transformers:", transformers.__version__)
print("tokenizers:", tokenizers.__version__)
print("numpy:", numpy.__version__)
print("cuda:", torch.cuda.is_available())
print("vllm:", getattr(vllm, "__file__", None))
assert torch.cuda.is_available()
assert triton.__version__.startswith("3.5.0")
assert transformers.__version__ == "4.56.0"
assert tokenizers.__version__ == "0.22.2"
assert numpy.__version__ == "2.2.6"
assert getattr(vllm, "__file__", None) is not None
PY}));

    run(q{nm -D } . shell_quote($vllm_so) . q{ | c++filt | grep -i cutlass_moe_mm_sm100 | grep ' T '});

    run(env_cmd($install, q{python -m vllm.entrypoints.openai.api_server --help >/dev/null}));

    print "[PASS] api_server --help works\n";
    print "[PASS] cutlass_moe_mm_sm100 is defined\n";
    print "[PASS] environment restored\n";
}

sub show_current_state {
    my ($install) = @_;

    section("Current state");

    run_allow_fail(env_cmd($install, q{python - <<'PY'
mods = ["torch", "triton", "transformers", "tokenizers", "numpy", "flashinfer", "vllm"]
for m in mods:
    try:
        mod = __import__(m)
        ver = getattr(mod, "__version__", "no __version__")
        path = getattr(mod, "__file__", None)
        print(f"{m}: {ver} | {path}")
    except Exception as e:
        print(f"{m}: ERROR | {e}")
PY}));
}

sub env_cmd {
    my ($install, $inner) = @_;
    my $activate = venv_activate($install);
    my $vllm_dir = vllm_src_dir($install);

    return q{bash -lc } . shell_quote(
        q{source } . shell_quote($activate) .
        q{ && cd } . shell_quote($vllm_dir) .
        q{ && export PYTHONPATH=} . shell_quote($vllm_dir) . q{:${PYTHONPATH:-}} .
        q{ && } . $inner
    );
}

sub venv_activate {
    my ($install) = @_;
    return File::Spec->catfile($install, '.vllm', 'bin', 'activate');
}

sub venv_python {
    my ($install) = @_;
    return File::Spec->catfile($install, '.vllm', 'bin', 'python');
}

sub vllm_src_dir {
    my ($install) = @_;
    return File::Spec->catdir($install, 'vllm');
}

sub triton_src_dir {
    my ($install) = @_;
    return File::Spec->catdir($install, 'triton');
}

sub vllm_so_path {
    my ($install) = @_;
    return File::Spec->catfile($install, 'vllm', 'vllm', '_C.abi3.so');
}

sub section {
    my ($title) = @_;
    print "\n========== $title ==========\n";
}

sub run {
    my ($cmd) = @_;
    print "\n\$ $cmd\n";

    if ($opt{dry_run}) {
        print "[DRY-RUN] skipped\n";
        return;
    }

    my $rc = system($cmd);
    if ($rc != 0) {
        my $exit = $rc >> 8;
        die "[FAIL] Command failed with exit code $exit\n";
    }
}

sub run_allow_fail {
    my ($cmd) = @_;
    print "\n\$ $cmd\n";

    return if $opt{dry_run};

    system($cmd);
}

sub check_file {
    my ($path, $msg) = @_;
    die "[FAIL] $msg\n" unless -f $path;
}

sub check_dir {
    my ($path, $msg) = @_;
    die "[FAIL] $msg\n" unless -d $path;
}

sub shell_quote {
    my ($s) = @_;
    $s =~ s/'/'"'"'/g;
    return "'$s'";
}

sub slurp {
    my ($path) = @_;
    open my $fh, '<', $path or return '';
    local $/ = undef;
    my $txt = <$fh>;
    close $fh;
    return defined $txt ? $txt : '';
}

sub usage {
    return <<"USAGE";
Usage:
  perl restore_vllm_from_snapshot.pl [options]

Options:
  --install-dir PATH
      vLLM installation path.
      Default: /local_opt/vllm-install

  --info-dir PATH
      Saved workable info path.
      Default: /local_opt/workable_llm_info

  --dry-run
      Print commands without executing them.

  --help
      Show this help.

Examples:
  perl restore_vllm_from_snapshot.pl

  perl restore_vllm_from_snapshot.pl \\
    --install-dir /local_opt/vllm-install \\
    --info-dir /local_opt/workable_llm_info

  perl restore_vllm_from_snapshot.pl --dry-run
USAGE
}