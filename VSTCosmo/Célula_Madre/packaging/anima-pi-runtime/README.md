# ANIMA Pi Runtime Debian Package

Empaquetado `.deb` de ANIMA para Raspberry Pi / Ubuntu ARM64.

Objetivo:

- Instalar todos los organelos como potencialidades latentes.
- Crear una instalación limpia de un organismo ANIMA local, sin depender del Rode/Mac.
- Usar audio local del sistema por defecto y silencio basal como estado válido.
- Crear identidad persistente `~/.anima/identity.json` con clave Ed25519.
- Correr nativo con `systemd --user`, sin Docker como servidor visible.
- Preservar configuración en `/etc/anima` y datos en `/var/lib/anima`.

Construcción local desde el Mac:

```bash
cd /Users/alexis/Desktop/RMD/Cosmolab/VSTCosmo/Célula_Madre
VERSION=0.2.3-dev packaging/anima-pi-runtime/build_deb_local.sh
```

Construcción desde el Mac usando una Pi como builder ARM64:

```bash
cd /Users/alexis/Desktop/RMD/Cosmolab/VSTCosmo/Célula_Madre
VERSION=0.2.3-dev PI_HOST=ubuntu@192.168.86.36 \
  packaging/anima-pi-runtime/build_deb_on_pi.sh
```

Salida:

```text
dist/anima-pi-runtime_0.2.3-dev_arm64.deb
```

Instalación en una Pi nueva:

```bash
sudo apt install ./anima-pi-runtime_0.2.3-dev_arm64.deb
```

Durante la instalación, si hay terminal interactiva, el paquete pregunta el nombre propio
del animalito y lo guarda en `/etc/anima/organismo.env`.

Para instalaciones no interactivas:

```bash
sudo ANIMA_NOMBRE="Nido" apt install ./anima-pi-runtime_0.2.3-dev_arm64.deb
```

Para cambiarlo después:

```bash
sudo anima setup --perfil limpio --nombre "Nido"
```

Upgrade:

```bash
sudo apt install ./anima-pi-runtime_VERSION_arm64.deb
```

La configuración vive en `/etc/anima` y se genera desde plantillas de
`/usr/share/anima-pi-runtime/config`. La identidad persistente vive en
`~/.anima/identity.json`. El nombre no se vuelve a preguntar si existe
`/etc/anima/identity.configured`.

Rollback básico:

```bash
sudo apt install ./anima-pi-runtime_VERSION_ANTERIOR_arm64.deb
sudo anima restart
```

Comandos:

```bash
anima status
anima start
anima stop
anima restart
anima cabeza rotar 0
anima cabeza rotar 180
anima cabeza vnc
```

Notas de v0.2.3-dev:

- Asume usuario `ubuntu`.
- Crea compatibilidad con `/home/ubuntu/anima/celula_madre` mediante symlink a `/opt/anima/celula_madre`.
- Los servicios se instalan como unidades de usuario en `~ubuntu/.config/systemd/user`.
- La pantalla SPI/fb1 y su VNC se activan si el hardware está presente.
- No toca al Organismo E salvo que se instale explícitamente en su Pi.
- Por defecto usa `ANIMA_AUDIO_MODE=local`, `VST_DISABLE_DIRECT_AUDIO=0`, `ANIMA_SDR_ENABLE=0` y `ANIMA_VISIBILITY=local`.
- `organelos.yml` y `hardware.yml` se sincronizan desde plantillas en upgrades; si existían versiones distintas, quedan respaldadas como `.bak.YYYYmmddHHMMSS`.
