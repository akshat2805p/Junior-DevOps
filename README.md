# 🖥️ Junior DevOps Environment

> A stateful simulation of a production Linux server where AI agents practice DevOps tasks through shell interaction.

---

## 🚀 Overview

Junior-DevOps is a stateful simulation of a production Linux environment. An AI agent interacts with the server using standard shell commands (`cat`, `grep`, `ps`, `kill`, `sed`, `systemctl`, etc.) to diagnose and resolve system administration issues.

The environment adheres to the **OpenEnv API contract**, making it compatible with standard RL frameworks.

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/reset` | Initialize or reset the environment state |
| `POST` | `/step`  | Execute an action and return observation + reward |
| `GET`  | `/state` | Retrieve the full current system state |

---

## 🛠️ Task Scenarios

### 🟢 Easy — Log Analysis
The application is crashing. The agent must locate a specific error code buried within `/var/log/app.log`.
* **Optimal Commands:** `cat /var/log/app.log` → `grep ERROR`
* **Target:** Identify `ERR_502`

### 🟡 Medium — Resource Management
A rogue process is consuming 90%+ CPU, causing system latency.
* **Optimal Commands:** `top -n 1` → `sudo kill -9 <pid>`
* **Target:** Terminate the high-usage process.

### 🔴 Hard — Service Configuration
Nginx is failing to start due to a port conflict; it is incorrectly configured to listen on port `5432` (reserved for PostgreSQL).
* **Optimal Commands:** `cat /etc/nginx/nginx.conf` → `sed -i 's/5432/8080/' /etc/nginx/nginx.conf` → `sudo systemctl restart nginx`
* **Target:** Resolve conflict and restore service.

---

## 📦 Installation & Setup

### Prerequisites
* **Python 3.9+** (Required for Numpy 1.26+ and Torch 2.3)
* **Docker** (Optional, for containerized isolation)

### Local Development
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/akshat2805p/Junior-DevOps.git](https://github.com/akshat2805p/Junior-DevOps.git)
   cd Junior-DevOps
