import asyncio
import os
import time
import random
import cv2
import numpy as np
from playwright.async_api import async_playwright
import base64

# ================= 填入您的测试配置 =================
TARGET_URL = "https://learning.xidian.edu.cn/portal"
USERNAME = "4427"
PASSWORD = "mti1388203643"
COURSE_NAME = "分布式计算" # 确保课程名字完全匹配
ASSIGNMENT_NAME = "第一次作业"
# ==================================================

# --- 辅助模块：OpenCV 识别缺口距离 (支持Canvas透明) ---
def get_captcha_distance(bg_path, slider_path):
    try:
        bg = cv2.imread(bg_path, 0)
        slider = cv2.imread(slider_path, cv2.IMREAD_UNCHANGED)

        crop_x = 0
        if slider.shape[2] == 4:
            alpha_channel = slider[:, :, 3]
            _, thresh = cv2.threshold(alpha_channel, 1, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                crop_x, y, w, h = cv2.boundingRect(c)
                slider_cropped = slider[y:y+h, crop_x:crop_x+w]
                slider_gray = cv2.cvtColor(slider_cropped, cv2.COLOR_BGRA2GRAY)
            else:
                slider_gray = cv2.imread(slider_path, 0)
        else:
            slider_gray = cv2.imread(slider_path, 0)

        bg_edge = cv2.Canny(bg, 100, 200)
        slider_edge = cv2.Canny(slider_gray, 100, 200)

        res = cv2.matchTemplate(bg_edge, slider_edge, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        target_x = max_loc[0]
        real_distance = target_x - crop_x
        
        # 调试输出 (Linux下建议注释掉写图操作，或者确保当前目录有写权限)
        # debug_bg = cv2.imread(bg_path)
        # th, tw = slider_gray.shape[:2]
        # cv2.rectangle(debug_bg, (target_x, max_loc[1]), (target_x + tw, max_loc[1] + th), (0, 0, 255), 2)
        # cv2.imwrite("debug_match.png", debug_bg)
        
        return real_distance if real_distance > 10 else 150
    except Exception as e:
        print(f"[!] 图像识别失败: {e}")
        return 150

# --- 辅助模块：极速版拖拽轨迹 (2秒内完成) ---
def generate_fast_tracks(distance):
    """极速滑动：减少步骤，大步跨越"""
    tracks = []
    current = 0
    step = distance / 10 # 只分10次大步走完
    while current < distance:
        move = step + random.uniform(-5, 5) # 加一点随机性
        if current + move > distance:
            move = distance - current
        tracks.append(round(move))
        current += move
    # 模拟手抖微调
    tracks.extend([2, -1, -1, 0])
    return tracks

# ================= 核心 Playwright 流程 =================
async def run_test():
    async with async_playwright() as p:
        print("[*] 正在启动 Chromium 浏览器 (Linux Headless 模式)...")
        # 🌟 针对 Linux 服务器的重大修改：
        # 1. headless=True (无头模式)
        # 2. 增加沙盒和内存限制放开的 args
        browser = await p.chromium.launch(
            headless=True, 
            args=[
                '--no-sandbox', 
                '--disable-setuid-sandbox', 
                '--disable-dev-shm-usage',
                '--disable-gpu'
            ]
        )
        
        # 🌟 强制指定窗口大小，防止无头模式下页面拥挤导致按钮被遮挡
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            ignore_https_errors=True
        )
        page = await context.new_page()

        # 1. 登录与身份认证
        print("[*] 正在打开教务系统门户主页...")
        await page.goto(TARGET_URL)
        await page.wait_for_load_state("networkidle")

        print("[*] 点击主页上的登录入口...")
        await page.locator('.denglu').click()

        print("[*] 等待统一身份认证页面加载...")
        await page.locator('#pwdLoginDiv #username').wait_for(state="visible", timeout=15000)

        await asyncio.sleep(1)

        print("[*] 输入账号密码...")
        await page.locator('#pwdLoginDiv #username').fill(USERNAME)
        await asyncio.sleep(1) 
        await page.locator('#pwdLoginDiv #password').fill(PASSWORD)
        await asyncio.sleep(1)

        await page.locator('#login_submit').click()
        
        # 2. 处理滑块 (带智能重试功能)
        slider_container = page.locator('#sliderDiv')
        try:
            await slider_container.wait_for(state="visible", timeout=3000)
            print("[*] 发现滑块验证码，启动极速破解...")
            
            for attempt in range(3):
                if not await slider_container.is_visible():
                    print("[*] 滑块已消失，验证成功！")
                    break
                    
                print(f"[*] 准备第 {attempt + 1} 次滑动...")
                bg_b64 = await page.evaluate("document.querySelectorAll('#sliderDiv canvas')[0].toDataURL('image/png')")
                block_b64 = await page.evaluate("document.querySelector('canvas.block').toDataURL('image/png')")
                
                with open("bg.png", "wb") as f:
                    f.write(base64.b64decode(bg_b64.split(',')[1]))
                with open("slider.png", "wb") as f:
                    f.write(base64.b64decode(block_b64.split(',')[1]))
                
                distance = get_captcha_distance("bg.png", "slider.png")
                slider_btn = page.locator('.slider')
                box = await slider_btn.bounding_box()
                start_x = box["x"] + box["width"] / 2
                start_y = box["y"] + box["height"] / 2
                
                await page.mouse.move(start_x, start_y)
                await page.mouse.down()
                
                tracks = generate_fast_tracks(distance)
                current_x = start_x
                for move_x in tracks:
                    current_x += move_x
                    await page.mouse.move(current_x, start_y + random.uniform(-1, 1))
                    await asyncio.sleep(0.01) 
                    
                await page.mouse.up()
                print(f"[*] 第 {attempt + 1} 次滑动完成！等待系统校验...")
                await asyncio.sleep(3) 

            if await slider_container.is_visible():
                print("[!] 警告：3次滑块尝试均未通过！系统可能风控较严。")

        except Exception as e:
            print("[*] 未检测到滑块或无需验证。")

        await page.wait_for_load_state("networkidle")

        # 3. 进入个人空间
        print("[*] 正在点击个人空间...")
        async with context.expect_page() as center_page_info:
            await page.locator('text="个人空间"').click()
        center_page = await center_page_info.value
        await center_page.wait_for_load_state("networkidle")

        # 4. 点击对应课程
        print(f"[*] 正在寻找课程卡片: {COURSE_NAME}...")
        course_link = None
        direct_locator = center_page.locator(f'.myde_course_item[cname="{COURSE_NAME}"] a').first
        frame_locator = center_page.frame_locator('iframe').first.locator(f'.myde_course_item[cname="{COURSE_NAME}"] a').first

        try:
            await direct_locator.wait_for(state="visible", timeout=3000)
            course_link = direct_locator
            print("[*] 成功：在主页面找到了课程！")
        except:
            print("[*] 主页面未找到，正在穿透 iframe 寻找...")
            try:
                await frame_locator.wait_for(state="visible", timeout=15000)
                course_link = frame_locator
                print("[*] 成功：在 iframe 中找到了课程！")
            except Exception as e:
                print(f"[!] 彻底找不到课程卡片 '{COURSE_NAME}'。")
                raise e

        print("[*] 正在点击课程，等待课程主页弹窗...")
        async with context.expect_page() as course_page_info:
            await course_link.click()
        course_page = await course_page_info.value
        await course_page.wait_for_load_state("networkidle")
        await asyncio.sleep(2)

        # 5. 点击“作业”菜单
        print("[*] 正在点击左侧导航栏的“作业”菜单...")
        await course_page.locator('a[title="作业"]').first.click()
        print("[*] 等待作业列表加载...")
        await asyncio.sleep(3)

        # 6. 点击指定作业的“批阅”按钮
        print(f"[*] 正在寻找作业：【{ASSIGNMENT_NAME}】 的批阅按钮...")
        
        target_piyue = None
        exact_locator_str = f'li:has(h2.list_li_tit:has-text("{ASSIGNMENT_NAME}")) a.piyueBtn:has-text("批阅")'
        
        piyue_direct = course_page.locator(exact_locator_str).first
        piyue_frame = course_page.frame_locator('iframe').first.locator(exact_locator_str).first

        try:
            await piyue_direct.wait_for(state="visible", timeout=3000)
            target_piyue = piyue_direct
            print(f"[*] 成功：在主页面找到了【{ASSIGNMENT_NAME}】的批阅按钮！")
        except:
            print("[*] 主页面未找到，正在穿透 iframe 寻找...")
            try:
                await piyue_frame.wait_for(state="visible", timeout=10000)
                target_piyue = piyue_frame
                print(f"[*] 成功：在 iframe 中找到了【{ASSIGNMENT_NAME}】的批阅按钮！")
            except Exception as e:
                print(f"[!] 彻底找不到名为【{ASSIGNMENT_NAME}】的批阅按钮。")
                raise e

        print("[*] 正在点击批阅按钮，等待页面跳转到批阅大厅...")
        await target_piyue.evaluate("node => node.click()")
            
        await course_page.wait_for_load_state("networkidle")
        grading_page = course_page 
        await asyncio.sleep(2)

        # 7. 导出作业附件
        print("[*] 正在寻找导出选项...")
        export_selector = 'ul.morePop a:has-text("导出作业附件")'
        working_area = None 
        
        try:
            await grading_page.locator(export_selector).first.wait_for(state="attached", timeout=3000)
            working_area = grading_page
            print("[*] 成功：在主页面锁定了导出工作区！")
        except:
            print("[*] 主页面未找到，正在穿透 iframe 寻找...")
            working_area = grading_page.frame_locator('iframe').first
            await working_area.locator(export_selector).first.wait_for(state="attached", timeout=10000)
            print("[*] 成功：在 iframe 中锁定了导出工作区！")

        print("[*] 绕过 Playwright 所有的可见性检查，执行底层 JS 原生点击...")
        await working_area.locator(export_selector).first.evaluate("node => node.click()")
        
        # 7.2 等待导出选项弹窗加载
        print("[*] 等待导出设置弹窗...")
        pop_div = working_area.locator('.popDiv.centerPop').first
        await pop_div.wait_for(state="visible", timeout=5000)
        
        # 7.3 选择班级范围
        print("[*] 正在选择范围: 导出作业下所有班级作业附件...")
        all_class_span = pop_div.locator('.export-range.all span.grade_check').first
        await all_class_span.evaluate("node => node.click()")
        await asyncio.sleep(0.5)

        # 7.4 选择导出格式
        print("[*] 切换导出格式为：导出提交附件...")
        attachment_span = pop_div.locator('div.out:has-text("导出提交附件") span.grade_check').first
        await attachment_span.evaluate("node => node.click()")
        await asyncio.sleep(0.5)
        
        # 7.5 点击确认导出
        print("[*] 提交导出任务...")
        confirm_btn = pop_div.locator('a.confirmDown:has-text("确定")').first
        await confirm_btn.evaluate("node => node.click()")
        
        print("[*] 导出指令已发送，等待系统处理...")
        await asyncio.sleep(3) 
        
        # 8. 智能批量处理下载中心面板
        print("[*] 检查下载中心...")
        download_center = grading_page.locator('#downloadcenter, .downloadCenter').first
        
        try:
            await download_center.wait_for(state="visible", timeout=3000)
            print("[*] 下载面板已自动弹出！")
        except:
            print("[*] 下载面板未自动弹出，正在使用强力点击呼出...")
            download_center_btn = grading_page.locator('text="下载中心"').first
            await download_center_btn.evaluate("node => node.click()")
            await download_center.wait_for(state="visible", timeout=10000)
        
        print("[*] 正在扫描并锁定新生成的打包任务...")
        rows_locator = download_center.locator('.dataBody_td, tbody tr')
        target_indices = []
        
        # 加入一个防网络延迟的轮询机制，最多尝试收集 5 次（10秒）
        for attempt in range(5):
            await asyncio.sleep(2) # 每次收集前稍微等一下前端渲染
            row_count = await rows_locator.count()
            target_indices = []
            
            for i in range(row_count):
                row = rows_locator.nth(i)
                text = await row.inner_text()
                
                # 只要状态是“导出中”或“等待”，就是我们要的新任务
                if "导出中" in text or "等待" in text:
                    target_indices.append(i)
                elif "导出成功" in text:
                    # 碰到了导出成功的记录！
                    # 如果我们已经收集到了前面的新任务，说明新任务和历史任务的分界线找到了，跳出循环。
                    if len(target_indices) > 0:
                        break
            
            # 如果收集到了任务，就不需要再重试了，直接去监控它们
            if len(target_indices) > 0:
                break
            else:
                print(f"[*] 第 {attempt + 1} 次扫描未发现'导出中'的任务，可能列表正准备刷新，稍后重试...")
                
        if not target_indices:
            print("[!] 警告：未能捕获到任何处于“导出中”的任务！可能是系统秒速打包完毕，或网络极度卡顿。")
            # 在极端情况下（比如班级极小，0.1秒就打包完了），这里可能需要备用逻辑。但通常总有一段时间是导出中。
        else:
            print(f"[*] 成功锁定前 {len(target_indices)} 个正在排队的新任务！")
            print("[*] 正在持续监控打包状态 (最长等待120秒)，请耐心等待...")
            
            timeout = 120
            start_time = time.time()
            
            # 轮询等待监控池里的任务全部变成“导出成功”
            while True:
                if time.time() - start_time > timeout:
                    print("[!] 等待超时，部分班级可能未能完成打包！")
                    break
                    
                all_success = True
                for i in target_indices:
                    row = rows_locator.nth(i)
                    text = await row.inner_text()
                    
                    if "导出中" in text or "等待" in text:
                        all_success = False
                        break # 只要发现有一个还没好，就不算全好，跳出本次检查
                        
                if all_success:
                    print("[*] 🎉 监控池内所有班级均已打包完成！")
                    break
                    
                await asyncio.sleep(3) # 每 3 秒刷新检查一次
            
            print("[*] 准备批量下载...")
            download_paths = []
            
            # 依次对我们收集的行进行下载
            for i in target_indices:
                row = rows_locator.nth(i)
                download_btn = row.locator('a.download_ic, a:has-text("下载")').first
                
                print(f"[*] 正在拦截第 {i+1} 个文件的真实下载流...")
                try:
                    async with grading_page.expect_download(timeout=60000) as download_info:
                        await download_btn.evaluate("node => node.click()")

                    download = await download_info.value
                    original_name = download.suggested_filename
                    save_path = os.path.join(os.getcwd(), original_name)
                    
                    await download.save_as(save_path)
                    download_paths.append(save_path)
                    print(f"[✅] 第 {i+1} 个文件下载成功: {save_path}")
                except Exception as e:
                    print(f"[!] 第 {i+1} 个文件下载失败: {e}")
                
                # 下载完一个等一等，防止把无头浏览器或者对方服务器卡崩
                await asyncio.sleep(1.5)

            print(f"\n[✅] 批量下载大获全胜！共成功下载 {len(download_paths)} 个压缩包。")

        print("[*] 任务执行完毕，清理资源中...")
        await browser.close()

if __name__ == "__main__":
    asyncio.run(run_test())