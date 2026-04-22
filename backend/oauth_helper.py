"""
Codex Team OAuth 认证辅助脚本
用法: python oauth_helper.py
"""
import os
import sys
import time
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading

# OAuth 配置
CLIENT_ID = "codex-team-oauth"
REDIRECT_URI = "http://localhost:8888/callback"
AUTH_URL = "https://codex.ai/oauth/authorize"
TOKEN_URL = "https://codex.ai/oauth/token"

# 本地服务器
HOST = "localhost"
PORT = 8888

class OAuthHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/callback":
            query = parse_qs(parsed.query)
            code = query.get("code", [None])[0]
            if code:
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(b"""
                    <html><head><title>授权成功</title></head>
                    <body style="font-family: sans-serif; padding: 40px; text-align: center;">
                        <h2>✅ 授权成功！</h2>
                        <p>请返回终端查看 token</p>
                        <p>Codex Team OAuth 授权码已获取，请复制上方显示的 code 并手动完成 token 交换。</p>
                    </body></html>
                """)
                print(f"\n✅ 获取到授权码: {code}")
                print("\n请将授权码粘贴到下方完成 token 交换:")
                print("(或者你可以直接使用这个 code 来换取 access_token)\n")
                self.server.code = code
                threading.Thread(target=lambda: time.sleep(3) and os._exit(0)).start()
            else:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Missing code parameter")
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        pass  # 静默日志

def main():
    print("=" * 60)
    print("Codex Team OAuth 认证助手")
    print("=" * 60)
    
    # 构建授权 URL
    params = {
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": "api"
    }
    
    auth_url = f"{AUTH_URL}?" + "&".join(f"{k}={v}" for k, v in params.items())
    
    print(f"\n1. 正在打开浏览器进行授权...")
    print(f"   如果浏览器没有自动打开，请手动访问:")
    print(f"   {auth_url}\n")
    
    webbrowser.open(auth_url)
    
    # 启动本地服务器
    print(f"2. 启动本地回调服务器 on http://{HOST}:{PORT}...\n")
    
    server = HTTPServer((HOST, PORT), OAuthHandler)
    server.code = None
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()