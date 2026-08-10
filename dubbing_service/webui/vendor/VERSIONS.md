# WebUI vendored dependencies

These browser assets are checked into the project so the production UI can start without public CDN access.

| Dependency | Version | Source | SHA-256 |
| --- | --- | --- | --- |
| Vue global production build | 3.5.39 | npm `vue@3.5.39`, `dist/vue.global.prod.js` | `349cb8e0ede3449b4962458daa03418518af50b4a9959a552cce35a8bec92805` |
| WaveSurfer core | 7.12.10 | npm `wavesurfer.js@7.12.10`, `dist/wavesurfer.min.js` | `7d4468fbd852b417d832428bf8e59ea7c40704604938e89417e21cdf179f978b` |
| WaveSurfer Regions plugin | 7.12.10 | npm `wavesurfer.js@7.12.10`, `dist/plugins/regions.min.js` | `ef5f152c453d6b95451502d95f813e168ba95977e7ff8724cff9af8311ffbfd3` |
| Tailwind precompiled CSS | 3.4.17 | npm `tailwindcss@3.4.17`, built from `src/tailwind.css` and `tailwind.config.cjs` | `682fe0d68473e5ec8004b6d82da26f77c25737b59d10b98d161cb617bd86bf35` |

The corresponding upstream license is stored beside each dependency. Update the version, asset, license, hash, and WebUI path together.
