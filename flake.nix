{
  description = "Voxtral Realtime 4B Pure C inference (voxtral.c)";

  inputs.flake-utils.url = "github:numtide/flake-utils";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        lib = pkgs.lib;

        isDarwin = pkgs.stdenv.isDarwin;
        isAarch64 = pkgs.stdenv.isAarch64;

        buildTarget = if isDarwin && isAarch64 then "mps" else "blas";

        darwinFrameworks = lib.optionals isDarwin ([
          pkgs.darwin.apple_sdk.frameworks.Accelerate
          pkgs.darwin.apple_sdk.frameworks.AudioToolbox
          pkgs.darwin.apple_sdk.frameworks.CoreFoundation
        ] ++ lib.optionals isAarch64 [
          pkgs.darwin.apple_sdk.frameworks.Foundation
          pkgs.darwin.apple_sdk.frameworks.Metal
          pkgs.darwin.apple_sdk.frameworks.MetalPerformanceShaders
          pkgs.darwin.apple_sdk.frameworks.MetalPerformanceShadersGraph
        ]);
      in {
        packages.default = pkgs.stdenv.mkDerivation {
          version = "unstable-2025-01-01";
          meta.mainPackage = "voxtral";
          pname = "voxtral";
          src = ./.;

          nativeBuildInputs = lib.optionals (isDarwin && isAarch64) [ pkgs.xxd ];
          buildInputs = lib.optionals (!isDarwin) [ pkgs.openblas ] ++ darwinFrameworks;

          buildPhase = ''
            make ${buildTarget} inspect
          '';

          installPhase = ''
            install -d "$out/bin"
            install -m 755 voxtral "$out/bin/voxtral"
            if [ -f inspect_weights ]; then
              install -m 755 inspect_weights "$out/bin/inspect_weights"
            fi
          '';
        };

        apps.default = {
          type = "app";
          program = lib.getExe self.packages.${system}.default;
        };

        devShells.default = pkgs.mkShell {
          inputsFrom = [ self.packages.${system}.default ];
        };
      });
}
