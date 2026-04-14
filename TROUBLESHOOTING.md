# 故障排查记录

## ETF数据更新失败 (akshare代理错误)

### 问题现象
```
ProxyError: HTTPSConnectionPool(host='88.push2.eastmoney.com', port=443): 
Max retries exceeded... RemoteDisconnected('Remote end closed connection without response')
```

### 根本原因
- `config/scheduler.py` 使用 `ak.fund_etf_hist_em()` 获取ETF数据
- 该接口依赖东方财富 `88.push2.eastmoney.com` 域名
- 公司网络环境下该域名被防火墙阻断，即使挂代理也无法访问

### 解决方案
**切换到新浪数据源 `ak.fund_etf_hist_sina()`**

新浪接口特点：
- 数据源：`sina.com.cn`
- 无需代理，可直接访问
- 字段格式：`date, open, high, low, close, volume`（与数据库表结构兼容）

### 代码变更 (config/scheduler.py)

```python
# 旧代码（东方财富 - 被阻断）
df = ak.fund_etf_hist_em(symbol=symbol, period="daily", 
                          start_date=start_str, adjust="qfq")

# 新代码（新浪 - 可用）
df = ak.fund_etf_hist_sina(symbol=code)  # symbol格式: sh510500, sz159915
```

### 验证方法
```bash
# 直接测试接口
python3 -c "
import akshare as ak
df = ak.fund_etf_hist_sina(symbol='sh510500')
print(df.tail(3))
"
```

### 定时任务状态
```bash
curl http://localhost:8787/api/scheduler/status
# 确认 running: true, jobs 包含 daily_etf_update
```

### 相关文件
- `config/scheduler.py` - ETF/股票数据更新调度器
- `web/app.py` - 定时任务注册（lifespan函数）

---

## 其他已知问题

### 代理环境变量
公司网络需要代理才能访问外网，但某些域名被阻断：
```bash
# 检查代理配置
echo $http_proxy  # http://127.0.0.1:7890
echo $https_proxy
```

### 解决方案优先级
1. **首选**：切换到可直连的数据源（如新浪）
2. **备选**：配置代理白名单绕过被阻断域名
3. **最后**：使用VPN等其他网络方案
