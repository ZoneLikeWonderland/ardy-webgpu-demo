# WebGPU 局域网 HTTPS 测试说明

## 为什么 HTTP IP 不能使用 WebGPU

WebGPU 的入口 `navigator.gpu` 只在 secure context 中暴露。浏览器把 `https://`、`http://127.0.0.1`、`http://localhost` 等 origin 视为 potentially trustworthy，但普通 HTTP 局域网地址（例如 `http://<SERVER_IP>:8766`）不在例外范围内。因此同一份页面通过 127.0.0.1 访问时可用，通过 <SERVER_IP> 的 HTTP 访问时 `window.isSecureContext=false` 且 `navigator.gpu` 不存在。

127.0.0.1 永远指向“运行浏览器的那台机器”，不是这台服务器。SSH `-L` 端口转发之所以有效，是因为浏览器看见的 origin 仍是本机 127.0.0.1，TCP 数据再由 SSH 隧道转到服务器。

## 当前可用入口

- 普通 HTTP（保留用于 localhost/SSH 隧道）：`http://<SERVER_IP>:8766/`
- 局域网 HTTPS 预检：`https://<SERVER_IP>:8767/`
- 局域网 HTTPS ARDY：`https://<SERVER_IP>:8767/infinite_demo.html`

8767 使用带 `IP:<SERVER_IP>` SAN 的测试证书，HTTPS 服务与原 8766 并行运行，未替换或中断原服务。

## 测试者使用方式

临时测试最简单的方式：

1. 打开 `https://<SERVER_IP>:8767/`。
2. 第一次会看到自签名证书警告，选择“高级”并继续访问。
3. 页面应显示“安全上下文：是”和“WebGPU API：可用”。
4. 再进入 infinite demo。

需要反复测试且不想看到证书警告时，可在受控测试机安装测试 CA：

1. 从 `http://<SERVER_IP>:8766/ardy-lan-ca.crt` 下载公开 CA 证书。
2. 核对 SHA-256 指纹：

   `14:83:D9:C7:43:00:EC:5A:C7:D0:DD:32:FA:39:95:76:BD:24:C4:22:34:19:E9:34:2B:0F:FE:95:75:41:27:6B`

3. Windows 双击证书，选择“安装证书”，放入“受信任的根证书颁发机构”。
4. 完全退出并重启 Edge/Chrome，再访问 8767。

只应在受控测试机器上信任该 CA；测试结束后可从 Windows 证书管理器删除 `ARDY WebGPU LAN Test CA`。公开 CA 证书可以分发，CA 私钥和服务器私钥不得分发。两份私钥保存在静态服务目录之外，权限为 600，HTTP/HTTPS 均无法下载。

## 其他方案

- 有 SSH 权限的测试者可以运行：

  `ssh -N -L 8766:127.0.0.1:8766 <user>@<SERVER_IP>`

  然后打开 `http://127.0.0.1:8766/`。

- Edge/Chrome 临时开发开关：在 `edge://flags/#unsafely-treat-insecure-origin-as-secure` 中加入 `http://<SERVER_IP>:8766` 并重启浏览器。该方式会人为降低浏览器安全约束，只适合临时测试。

- 正式、无警告地给更多人使用：提供一个域名和由公网 CA 或组织内网 CA 签发的证书，再把域名解析到该服务器。由于 `<SERVER_IP>` 是私网 IP，公网 CA 不会直接为这个 IP 签发普通可信证书。

## 服务端验证

- Python 3.11 成功加载证书链。
- TLS 1.3 协商成功，cipher 为 `TLS_AES_256_GCM_SHA384`。
- 使用测试 CA 对 `<SERVER_IP>` 做 IP 校验，`Verify return code: 0 (ok)`。
- HTTPS 预检页、infinite demo、CA 下载和 motion-stats API 均返回 HTTP 200。
- 没有安装任何软件包，也没有执行 conda 命令。

