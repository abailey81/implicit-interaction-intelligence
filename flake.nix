{
  description = "Implicit Interaction Intelligence (I3) — deterministic dev shell & package build via Nix Flakes";

  inputs = {
    nixpkgs.url      = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url  = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachSystem [
      "x86_64-linux"
      "x86_64-darwin"
      "aarch64-linux"
      "aarch64-darwin"
    ] (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        python = pkgs.python311;

        # Developer-time tools — present in both `nix develop` and the package build.
        commonTools = with pkgs; [
          python
          poetry
          uv
          ruff
          mypy
          pre-commit
          mkdocs
          just
          git
          docker
        ];

        # A thin runtime environment that holds the declared Python deps. For
        # a production-grade build we would use poetry2nix or uv2nix here — we
        # keep the flake minimal so `nix build` just produces a venv wheel.
        pythonEnv = python.withPackages (ps: with ps; [
          pip
          setuptools
          wheel
        ]);

      in {
        # -----------------------------------------------------------------
        # devShell — entered via `nix develop` or direnv (`use flake .`).
        # -----------------------------------------------------------------
        devShells.default = pkgs.mkShell {
          name = "i3-dev";

          packages = commonTools ++ [
            pythonEnv
            pkgs.stdenv.cc.cc.lib
            pkgs.zlib
          ];

          shellHook = ''
            echo ""
            echo "  I3 :: Nix dev shell ready."
            echo "  python: $(python --version)"
            echo "  uv:     $(uv --version 2>/dev/null || echo 'not installed')"
            echo "  poetry: $(poetry --version 2>/dev/null || echo 'not installed')"
            echo ""
            echo "  Try:    uv sync --all-extras   OR   poetry install --with dev"
            echo ""

            # Ensure uv caches land inside the repo, not $HOME, for hermeticity.
            export UV_CACHE_DIR="$PWD/.uv-cache"
            export UV_PYTHON_PREFERENCE="system"
          '';
        };

        # -----------------------------------------------------------------
        # packages.default — the I3 wheel, built hermetically.
        #
        # We shell out to uv inside a nix build sandbox. For a fully
        # poetry2nix / uv2nix integration see docs/operations/reproducibility.md.
        # -----------------------------------------------------------------
        packages.default = pkgs.stdenv.mkDerivation {
          pname   = "i3";
          version = "0.1.0";

          src = ./.;

          nativeBuildInputs = commonTools;

          buildPhase = ''
            export HOME=$TMPDIR
            export UV_CACHE_DIR=$TMPDIR/uv-cache
            uv build --wheel --out-dir dist
          '';

          installPhase = ''
            mkdir -p $out
            cp -r dist $out/
          '';

          meta = with pkgs.lib; {
            description = "Implicit Interaction Intelligence — behavioural-biometric auth";
            homepage    = "https://github.com/i3/implicit-interaction-intelligence";
            license     = licenses.asl20;
            platforms   = platforms.unix;
          };
        };

        # Convenience alias: `nix run .#bootstrap`
        apps.bootstrap = {
          type = "app";
          program = toString (pkgs.writeShellScript "i3-bootstrap" ''
            exec ${pkgs.bash}/bin/bash ${./scripts/uv_bootstrap.sh}
          '');
        };
      });
}
