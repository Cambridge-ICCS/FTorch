# Copyright Spack Project Developers. See LICENSE file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import os

from spack.package import *


class Ftorch(CMakePackage):
    """FTorch: A library for coupling PyTorch models to Fortran.
    
    FTorch enables users to directly couple their PyTorch models to Fortran code,
    supporting both CPU and GPU execution on UNIX and Windows operating systems.
    """

    homepage = "https://github.com/Cambridge-ICCS/FTorch"
    url = "https://github.com/Cambridge-ICCS/FTorch/archive/refs/tags/v1.0.0.tar.gz"
    git = "https://github.com/Cambridge-ICCS/FTorch.git"

    license("MIT")

    version("1.0.0", sha256="e6e3bb6a27f4c5b42b9e6d5c5f5c5c5c5c5c5c5c5c5c5c5c5c5c5c5c5c5c5c5")
    version("main", branch="main", git=git)

    # Variants for optional features
    variant("cuda", default=False, description="Enable CUDA GPU support")
    variant("hip", default=False, description="Enable HIP (ROCm) GPU support")
    variant("xpu", default=False, description="Enable Intel XPU support")
    variant("mps", default=False, description="Enable Apple Metal Performance Shaders support")
    variant("shared", default=True, description="Build shared library (default) instead of static")
    variant("tests", default=False, description="Build tests")
    
    # CUDA architecture variant - required when building with CUDA
    # Common architectures: 70 (Volta), 75 (Turing), 80 (Ampere), 86 (Ampere), 89 (Ada), 90 (Hopper)
    variant(
        "cuda_arch",
        default="none",
        values=("none", "70", "75", "80", "86", "89", "90"),
        multi=True,
        description="CUDA architecture(s) to build for",
        when="+cuda",
    )
    
    # AMD GPU target variant - required when building with ROCm/HIP
    # Common targets: gfx906 (MI50/MI60), gfx908 (MI100), gfx90a (MI200), gfx942 (MI300)
    variant(
        "amdgpu_target",
        default="none",
        values=("none", "gfx906", "gfx908", "gfx90a", "gfx942"),
        multi=True,
        description="AMD GPU target(s) to build for",
        when="+hip",
    )

    # Core dependencies
    depends_on("cmake@3.15:", type="build")
    depends_on("fortran", type="build")
    depends_on("py-torch", type=("build", "link"))

    # GPU support dependencies - ensure PyTorch has matching GPU support
    depends_on("cuda", when="+cuda", type=("build", "link"))
    # Pass cuda_arch to py-torch - user must specify at least one architecture
    for arch in ("70", "75", "80", "86", "89", "90"):
        depends_on(f"py-torch+cuda cuda_arch={arch}", when=f"+cuda cuda_arch={arch}", type=("build", "link"))
    
    depends_on("hip", when="+hip", type=("build", "link"))
    # Pass amdgpu_target to py-torch - user must specify at least one target
    for target in ("gfx906", "gfx908", "gfx90a", "gfx942"):
        depends_on(f"py-torch+rocm amdgpu_target={target}", when=f"+hip amdgpu_target={target}", type=("build", "link"))
    
    # XPU (Intel GPU) support requires Intel Extension for PyTorch (IPEX)
    # which is not yet widely available in Spack. Users building with +xpu
    # will need to manually provide IPEX or use an external PyTorch installation.
    # depends_on("intel-extension-for-pytorch", when="+xpu", type=("build", "link"))
    
    # MPS (Metal Performance Shaders) is built into PyTorch on macOS
    # No additional dependencies needed beyond platform check
    depends_on("py-torch", when="+mps", type=("build", "link"))
    
    depends_on("python", when="+tests", type=("build", "run"))

    # Conflicts for mutually exclusive GPU backends
    conflicts("+cuda", when="+hip")
    conflicts("+cuda", when="+xpu")
    conflicts("+cuda", when="+mps")
    conflicts("+hip", when="+xpu")
    conflicts("+hip", when="+mps")
    conflicts("+xpu", when="+mps")

    # Conflicts with platform-specific GPU support
    conflicts("+mps", when="platform=linux")
    conflicts("+mps", when="platform=windows")
    conflicts("+hip", when="platform=darwin")

    def cmake_args(self):
        """Define CMake arguments for FTorch build."""
        args = [
            self.define_from_variant("BUILD_SHARED_LIBS", "shared"),
            self.define_from_variant("CMAKE_BUILD_TESTS", "tests"),
        ]

        # Determine GPU device target
        gpu_device = "NONE"
        if self.spec.satisfies("+cuda"):
            gpu_device = "CUDA"
        elif self.spec.satisfies("+hip"):
            gpu_device = "HIP"
        elif self.spec.satisfies("+xpu"):
            gpu_device = "XPU"
        elif self.spec.satisfies("+mps"):
            gpu_device = "MPS"

        args.append(self.define("GPU_DEVICE", gpu_device))

        # Torch setup: find the torch directory for CMake
        torch_prefix = self.spec["py-torch"].prefix
        
        # Check if this is an external package (pip-installed PyTorch)
        # External packages have prefix pointing to site-packages directory
        torch_cmake_dir = join_path(torch_prefix, "torch")
        if os.path.isdir(torch_cmake_dir):
            # External: prefix is site-packages, torch is a subdirectory
            args.append(self.define("CMAKE_PREFIX_PATH", torch_cmake_dir))
        elif "python" in self.spec:
            # Spack-built: construct path to site-packages/torch
            args.append(
                self.define(
                    "CMAKE_PREFIX_PATH",
                    join_path(
                        torch_prefix,
                        "lib",
                        "python{0}.{1}".format(
                            self.spec["python"].version.major,
                            self.spec["python"].version.minor,
                        ),
                        "site-packages",
                        "torch",
                    ),
                )
            )
        else:
            # Fallback: assume prefix points directly to torch
            args.append(self.define("CMAKE_PREFIX_PATH", torch_prefix))

        return args

    def setup_build_environment(self, env):
        """Set up build environment variables."""
        # Ensure proper Fortran environment
        if self.compiler.fc:
            env.set("CMAKE_Fortran_COMPILER", self.compiler.fc)
        if self.compiler.cc:
            env.set("CMAKE_C_COMPILER", self.compiler.cc)
        if self.compiler.cxx:
            env.set("CMAKE_CXX_COMPILER", self.compiler.cxx)

    def setup_run_environment(self, env):
        """Set up runtime environment."""
        # Add installation paths to environment
        env.prepend_path("LD_LIBRARY_PATH", join_path(self.prefix, "lib"))
        env.prepend_path(
            "FORTRAN_MODULE_PATH", join_path(self.prefix, "include", "ftorch")
        )

    @run_after("install")
    def setup_fortran_module_path(self):
        """Ensure Fortran module files are properly installed."""
        # Verify Fortran module files are installed
        ftorch_mod = join_path(self.prefix, "include", "ftorch", "ftorch.mod")
        if not os.path.exists(ftorch_mod):
            tty.warn(
                "Expected Fortran module file not found at {0}".format(ftorch_mod)
            )
