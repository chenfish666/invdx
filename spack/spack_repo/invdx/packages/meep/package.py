# Copyright Spack Project Developers. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
#
# invdx 專案自帶 package repo（namespace invdx）：整份複製 upstream
# spack-packages 的 meep/package.py 後修改，理由見
# ~/.claude/plans/invdx-design-infra.md 第二輪 L2.1「第二段」——
# 不可用 class 繼承（父類 depends_on("python@:3.11") 只能收緊不能放寬）。
#
# 本檔相對 upstream 的差異（僅三處，皆有標註 invdx: 註解）：
#   1. 新增 version("1.34.0", ...)，url 顯式指向 GitHub release 資產
#      （而非預設猜測的 git tag 原始封存），理由見該行註解。
#   2. python 版本閘：@:1.31 沿用舊上限，@1.32: 放寬到 3.11-3.13
#      （上游 NEWS 稱 1.32.0 才修 Python 3.12+ 相容）。
#   3. py-numpy 版本閘：@1.32: 起要求 numpy 2，對齊 conda baseline。

from spack_repo.builtin.build_systems.autotools import AutotoolsPackage

from spack.package import *


class Meep(AutotoolsPackage):
    """Meep (or MEEP) is a free finite-difference time-domain (FDTD) simulation
    software package developed at MIT to model electromagnetic systems."""

    homepage = "http://ab-initio.mit.edu/wiki/index.php/Meep"
    git = "https://github.com/NanoComp/meep.git"
    url = "https://github.com/NanoComp/meep/archive/refs/tags/v1.21.0.tar.gz"

    license("GPL-2.0-or-later")

    version("master", branch="master")

    # invdx: bump 目標。sha256 為 `spack checksum meep 1.34.0` 算出之值（git tag
    # 原始封存，走 class url 樣板 archive/refs/tags/）。
    #
    # 供應鏈互驗記錄（2026-08-17）：conda-forge pymeep-feedstock meta.yaml
    # 記錄的是另一顆 sha256（3c9284…60bc6，來源 releases/download/ 的
    # `make dist` 產物）。兩顆 tarball 皆直接對 github.com 核實為 NanoComp/meep
    # 官方資產、內容皆為同一 tag v1.34.0，非供應鏈風險——差異純屬封裝：
    # release dist tarball 的 python/Makefile.am 之 EXTRA_DIST 未列
    # python/numpy.i（上游 dist 腳本的既有缺陷），導致該顆建置在 SWIG 產生
    # meep-python.cxx 前就因缺檔失敗（make: No rule to make target 'numpy.i'）；
    # git tag 封存含完整已追蹤檔案（含 numpy.i），本 recipe 既有的
    # --enable-maintainer-mode + autoconf/automake/libtool build deps
    # 正是為此封裝設計。故採 git tag 封存版本，與 upstream package.py 對其他
    # 版本的預設行為一致。
    version("1.34.0", sha256="1fa6dd4a363cd8085533e18913b02bba958618518c5843e94483545651d78ea4")

    version("1.29.0", sha256="f63bdf6a8fbae8aad87d4f683da3a466d687848a53bbebe1d6935fb268aeeffa")
    version("1.28.0", sha256="fe79ec9b0d0cf87c3855a1661a38f23a3100120174f7e2df8add96cafe201544")
    version("1.25.0", sha256="3e5d6c6ef69a8cc7810bdd6d681ae494bfe7a4e91041abe5494f5c8a82d02e6f")
    version("1.21.0", sha256="71911cd2f38b15bdafe9a27ad111f706f24717894d5f9b6f9f19c6c10a0d5896")
    version(
        "1.3",
        sha256="564c1ff1b413a3487cf81048a45deabfdac4243a1a37ce743f4fcf0c055fd438",
        url="http://ab-initio.mit.edu/meep/meep-1.3.tar.gz",
    )
    version(
        "1.2.1",
        sha256="f1f0683e5688d231f7dd1863939677148fc27a6744c03510e030c85d6c518ea5",
        url="http://ab-initio.mit.edu/meep/meep-1.2.1.tar.gz",
    )
    version(
        "1.1.1",
        sha256="7a97b5555da1f9ea2ec6eed5c45bd97bcd6ddbd54bdfc181f46c696dffc169f2",
        url="http://ab-initio.mit.edu/meep/old/meep-1.1.1.tar.gz",
    )

    variant("blas", default=True, description="Enable BLAS support")
    variant("lapack", default=True, description="Enable LAPACK support")
    variant("harminv", default=True, description="Enable Harminv support")
    variant("guile", default=True, description="Enable Guilde support")
    variant("libctl", default=True, description="Enable libctl support")
    variant("mpi", default=True, description="Enable MPI support")
    variant("hdf5", default=True, description="Enable HDF5 support")
    variant("gsl", default=True, description="Enable GSL support")
    variant("python", default=True, description="Enable Python support")
    variant("single", default=False, description="Enable Single Precision")
    variant("libgdsii", default=True, description="Enable libGDSII support")
    variant("mpb", default=True, description="Enable MPB support")
    variant("openmp", default=True, description="Enable OpenMP support")

    depends_on("c", type="build")  # generated
    depends_on("cxx", type="build")  # generated
    depends_on("fortran", type="build")

    depends_on("autoconf", type="build", when="@1.21.0:")
    depends_on("automake", type="build", when="@1.21.0:")
    depends_on("libtool", type="build", when="@1.21.0:")

    depends_on("fftw-api")
    depends_on("blas", when="+blas")
    depends_on("lapack", when="+lapack")
    depends_on("harminv", when="+harminv")
    depends_on("guile@:2", when="@:1.4+guile")
    depends_on("guile@2:", when="@1.4:+guile")
    depends_on("libctl@3.2", when="@:1.3+libctl")
    depends_on("libctl@4:", when="+libctl")
    depends_on("mpi", when="+mpi")
    depends_on("hdf5~mpi", when="+hdf5~mpi")
    depends_on("hdf5+mpi", when="+hdf5+mpi")
    depends_on("gsl", when="+gsl")
    with when("+python"):
        # invdx: python 版本閘放寬。upstream 無條件釘 python@:3.11；
        # 上游 NEWS 稱 1.32.0 起才修好 Python 3.12+ 相容，故 @:1.31 沿用
        # 舊上限、@1.32: 放寬到 3.11-3.13（涵蓋 1.34.0）。
        depends_on("python@:3.11", when="@:1.31")
        depends_on("python@3.11:3.13", when="@1.32:")
        depends_on("py-numpy")
        # invdx: numpy 版本閘。1.32: 起要求 numpy 2，對齊 conda baseline
        # （pymeep 1.34.0 已驗證 numpy 2 世代）。
        depends_on("py-numpy@2:", when="@1.32:")
        depends_on("swig")
        depends_on("py-mpi4py", when="+mpi")
    depends_on("libgdsii", when="+libgdsii")
    depends_on("mpb", when="+mpb")

    def configure_args(self):
        spec = self.spec

        config_args = ["LDFLAGS={0}".format(spec["fftw-api"].libs.ld_flags)]

        config_args.append("--enable-shared")

        if "+blas" in spec:
            config_args.append("--with-blas={0}".format(spec["blas"].prefix.lib))
        else:
            config_args.append("--without-blas")

        if "+lapack" in spec:
            config_args.append("--with-lapack={0}".format(spec["lapack"].prefix.lib))
        else:
            config_args.append("--without-lapack")

        if "+libctl" in spec:
            config_args.append(
                "--with-libctl={0}".format(join_path(spec["libctl"].prefix.share, "libctl"))
            )
        else:
            config_args.append("--without-libctl")

        if "+python" in spec:
            config_args.append("--with-python")
        else:
            config_args.append("--without-python")
            config_args.append("--without-scheme")

        if "+single" in spec:
            config_args.append("--enable-single")

        config_args.extend(self.with_or_without("mpi"))
        config_args.extend(self.with_or_without("hdf5"))
        config_args.extend(self.with_or_without("openmp"))

        if spec.satisfies("@1.21.0:"):
            config_args.append("--enable-maintainer-mode")

        return config_args

    def check(self):
        spec = self.spec

        # aniso_disp test fails unless installed with harminv
        # near2far test fails unless installed with gsl
        if "+harminv" in spec and "+gsl" in spec:
            # Most tests fail when run in parallel
            # 2D_convergence tests still fails to converge for unknown reasons
            make("check", parallel=False)
