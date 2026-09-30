# 内网穿透 / 公网访问 (Cloudflare Quick Tunnel)

让别人**通过公网**访问你本机跑的 ChineseFoodNet Demo，**完全免费、不要域名、不要账号**。

我们用的是 **Cloudflare Quick Tunnel** —— 它会给你一个一次性的 `https://xxx.trycloudflare.com` 临时 URL，关掉就失效，最适合演示、远程展示、临时给同学/老师看 Demo。

---

## TL;DR (3 步上手)

```bash
# 1. 一次性下载 cloudflared (大约 30 MB，下到 food-web/bin/)
# Windows
scripts\install_cloudflared.bat
# Linux / macOS
scripts/install_cloudflared.sh

# 2. 启动 FastAPI 服务 (一个窗口)
scripts\run_dev.bat            # 或 scripts/run_dev.sh

# 3. 启动隧道 (另开一个窗口)
scripts\start_tunnel.bat       # 或 scripts/start_tunnel.sh
```

启动隧道后会看到类似下面的输出：

```
Your quick Tunnel has been created! Visit it at (NOTE: this URL is ephemeral):
https://crazy-little-words-here.trycloudflare.com
```

把这个 URL 发给任何人，他们就能在浏览器里打开你的网站，调你本机的模型。

---

## 工作原理

```
   你朋友的浏览器
        |
        v
 trycloudflare.com (Cloudflare 边缘节点)
        |
        v
 cloudflared 进程 (你本机出站长连接)
        |
        v
 http://localhost:8000  (FastAPI + 模型)
```

- `cloudflared` 在你本机**主动**出站连接到 Cloudflare 的边缘网络（HTTPS/443），所以**不需要在路由器开端口、不需要公网 IP**。
- 公网请求走 Cloudflare → 沿你那条长连接 → 回到 `localhost:8000`。
- Cloudflare 自动给一个 HTTPS 证书（trycloudflare.com 的子域），不用自己折腾 TLS。

---

## 注意事项

1. **URL 是临时的**。每次启动 `start_tunnel` 都会拿到**全新**的 trycloudflare.com 子域；关掉脚本，URL 立即失效。这是 quick tunnel 的特点（不要账号的代价）。如果你需要**固定**的 URL，看下面的 [稳定 URL](#想要固定-url) 一节。
2. **没有访问控制**。任何拿到 URL 的人都能用。Demo 期间发完就关。
3. **流量走你的本机带宽**。每张图片要从访问者 → Cloudflare → 你家上行带宽，所以家里上行慢的话推理体验也会慢。
4. **CORS 已经开 `*`** (见 `server.py` 的 `CORS_ORIGINS`)，所以走 trycloudflare.com 没有跨域问题。
5. **首次启动** cloudflared 大概要 1–3 秒才会打印出 URL，请耐心。

---

## 校验是否打通

URL 出来后，在自己电脑或手机的另一个浏览器里打开：

```
https://你拿到的子域.trycloudflare.com/api/health
```

应该返回 JSON，里面有 `"status":"ok"` 和 `"num_classes":208`。然后再访问根路径：

```
https://你拿到的子域.trycloudflare.com/
```

就能看到完整网页，可以上传图片测试。

---

## 想要固定 URL？

Quick Tunnel 每次都换地址。如果你想要**永久不变**的 URL，三种思路：

### A. Cloudflare Named Tunnel (推荐 / 仍然免费)
- 需要一个 **Cloudflare 账号** (免费)。
- 需要一个**自己的域名**（最便宜的 .xyz 几块钱一年；或把现成的免费 freenom 之类的接入）。
- 在 Cloudflare 上做一次 `cloudflared tunnel login`，拿到永久 token。
- 之后每次起隧道地址都一样：`https://demo.你的域名.com`。
- 操作指南：<https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/>

### B. ngrok (有限免费)
- 需要 ngrok 账号（免费）。
- 免费档每次仍然是随机子域（除非升级付费版才能锁定 `xxx.ngrok.app`）。
- 安装一行：[https://ngrok.com/download](https://ngrok.com/download)
- 用法：`ngrok http 8000`
- 适合：你已经有 ngrok 账号、不想换工具。

### C. 国内常见方案
- **frp / nps / Sakura Frp / 花生壳**：通常需要自己有一台公网 VPS 或买一个免费/廉价中转节点。配置略繁琐，但中国大陆访问体验更稳。
- **Tailscale Funnel**：免费、稳定、有固定子域，但需要注册 Tailscale 账号；只支持 1080/8443/443 等端口，要把 FastAPI 改到 8443。

---

## 故障排查

### 1. `bin\cloudflared.exe` 下载失败
- 在中国大陆直接访问 GitHub 经常超时。`install_cloudflared.bat` 已经做了 `ghproxy.com` 镜像兜底；如果**两个**都失败了，按提示手动下载，把 `cloudflared-windows-amd64.exe` 重命名为 `cloudflared.exe`，放到 `food-web\bin\` 下即可。
- 也可以试试：`winget install --id Cloudflare.cloudflared`，然后把 `cloudflared.exe` 拷到 `food-web\bin\`。

### 2. 看到 URL 但访问超时
- 确认 FastAPI 真的在 `http://localhost:8000` 起来了，浏览器本机访问 `http://localhost:8000` 必须能打开。
- Windows Defender / 防火墙可能拦 cloudflared，**第一次**启动会弹窗，选"允许"。

### 3. 朋友访问能打开网页但上传图片报错
- 检查 `server.py` 的 `MAX_UPLOAD_MB`，默认 10MB。手机直接拍的大图可能超过。
- F12 看 Network → `/api/predict` 的状态码：
  - `413` → 图片太大，调高 `MAX_UPLOAD_MB`
  - `415` → 上传的不是图片格式
  - `500` → 看本机 `uvicorn` 终端的报错栈

### 4. cloudflared 启动后立刻退出
- 多半是端口/网络被防火墙挡。试试用 `bin\cloudflared.exe tunnel --url http://localhost:8000 --loglevel debug` 看详细日志。

---

## 关停

- 先 `Ctrl+C` 关 `start_tunnel`，URL 立即作废。
- 再 `Ctrl+C` 关 `run_dev`，本机服务停掉。

整个过程**没有在你的机器上残留任何后台服务**，纯前台进程。

---

## 安全提示

- 这个 URL 上线期间，**全互联网**都可以请求你的 `/api/predict`，等于把你显卡/CPU 借出去给陌生人跑推理。Demo 完记得关。
- 真要做生产部署，请：
  1. 收紧 `CORS_ORIGINS` (改成你前端的真实域名)
  2. 加访问令牌 / Basic Auth / Cloudflare Access
  3. 用 named tunnel + 自己的域名 + Cloudflare WAF
