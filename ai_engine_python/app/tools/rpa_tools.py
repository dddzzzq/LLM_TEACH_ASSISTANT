"""
教务系统RPA工具模块 - 基于Playwright的自动化作业下载
"""
import asyncio
import os
import time
import random
import base64
import logging
from typing import List, Union
from pathlib import Path

try:
    import cv2
    import numpy as np
    from playwright.async_api import async_playwright
except ImportError as e:
    logging.warning(f"缺少必要的依赖库: {e}。请安装: pip install opencv-python numpy playwright")

# 配置日志
logger = logging.getLogger(__name__)

# 默认配置
DEFAULT_SAVE_DIR = "/root/autodl-tmp/dzq/homework"
TARGET_URL = "https://learning.xidian.edu.cn/portal"


def get_captcha_distance(bg_path: str, slider_path: str) -> int:
    """
    OpenCV识别滑块验证码缺口距离
    
    Args:
        bg_path: 背景图片路径
        slider_path: 滑块图片路径
    
    Returns:
        计算出的滑动距离
    """
    try:
        bg = cv2.imread(bg_path, 0)
        slider = cv2.imread(slider_path, cv2.IMREAD_UNCHANGED)
        
        if bg is None or slider is None:
            logger.error("无法读取验证码图片")
            return 150
        
        crop_x = 0
        if len(slider.shape) == 3 and slider.shape[2] == 4:
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
        
        if slider_gray is None:
            logger.error("无法处理滑块图片")
            return 150
        
        bg_edge = cv2.Canny(bg, 100, 200)
        slider_edge = cv2.Canny(slider_gray, 100, 200)
        
        res = cv2.matchTemplate(bg_edge, slider_edge, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        target_x = max_loc[0]
        real_distance = target_x - crop_x
        
        return real_distance if real_distance > 10 else 150
    except Exception as e:
        logger.error(f"图像识别失败: {e}")
        return 150


def generate_fast_tracks(distance: int) -> List[int]:
    """
    生成极速版拖拽轨迹（2秒内完成）
    
    Args:
        distance: 总滑动距离
    
    Returns:
        轨迹列表
    """
    tracks = []
    current = 0
    step = distance / 10  # 只分10次大步走完
    while current < distance:
        move = step + random.uniform(-5, 5)  # 加一点随机性
        if current + move > distance:
            move = distance - current
        tracks.append(round(move))
        current += move
    # 模拟手抖微调
    tracks.extend([2, -1, -1, 0])
    return tracks


async def fetch_homework_from_portal(
    username: str,
    password: str,
    course_name: str,
    assignment_name: str,
    save_dir: str = DEFAULT_SAVE_DIR
) -> Union[List[str], str]:
    """
    从教务系统下载作业附件（LangChain Tool）
    
    Args:
        username: 教务系统用户名
        password: 教务系统密码
        course_name: 课程名称（需完全匹配）
        assignment_name: 作业名称
        save_dir: 文件保存目录，默认为/tmp/ai_grade_homeworks
    
    Returns:
        成功时返回ZIP文件绝对路径列表，失败时返回错误信息字符串
    """
    logger.info(f"开始RPA抓取任务 - 用户: {username}, 课程: {course_name}, 作业: {assignment_name}")
    
    # 确保保存目录存在
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    download_paths = []
    
    try:
        async with async_playwright() as p:
            logger.info("正在启动 Chromium 浏览器 (Linux Headless 模式)...")
            # 针对 Linux 服务器的配置：无头模式 + 沙盒绕过
            browser = await p.chromium.launch(
                headless=True,
                args=[
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-gpu'
                ]
            )
            
            # 强制指定窗口大小，防止无头模式下页面拥挤
            context = await browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                ignore_https_errors=True
            )
            page = await context.new_page()
            
            # 1. 登录与身份认证
            logger.info("正在打开教务系统门户主页...")
            await page.goto(TARGET_URL)
            await page.wait_for_load_state("networkidle")
            
            logger.info("点击主页上的登录入口...")
            await page.locator('.denglu').click()
            
            logger.info("等待统一身份认证页面加载...")
            await page.locator('#pwdLoginDiv #username').wait_for(state="visible", timeout=15000)
            await asyncio.sleep(1)
            
            logger.info("输入账号密码...")
            await page.locator('#pwdLoginDiv #username').fill(username)
            await asyncio.sleep(1)
            await page.locator('#pwdLoginDiv #password').fill(password)
            await asyncio.sleep(1)
            
            await page.locator('#login_submit').click()
            
            # 2. 处理滑块验证码（带智能重试功能）
            slider_container = page.locator('#sliderDiv')
            try:
                await slider_container.wait_for(state="visible", timeout=3000)
                logger.info("发现滑块验证码，启动极速破解...")
                
                for attempt in range(3):
                    if not await slider_container.is_visible():
                        logger.info("滑块已消失，验证成功！")
                        break
                    
                    logger.info(f"准备第 {attempt + 1} 次滑动...")
                    bg_b64 = await page.evaluate(
                        "document.querySelectorAll('#sliderDiv canvas')[0].toDataURL('image/png')"
                    )
                    block_b64 = await page.evaluate(
                        "document.querySelector('canvas.block').toDataURL('image/png')"
                    )
                    
                    bg_path = os.path.join(save_dir, f"bg_{attempt}.png")
                    slider_path = os.path.join(save_dir, f"slider_{attempt}.png")
                    
                    with open(bg_path, "wb") as f:
                        f.write(base64.b64decode(bg_b64.split(',')[1]))
                    with open(slider_path, "wb") as f:
                        f.write(base64.b64decode(block_b64.split(',')[1]))
                    
                    distance = get_captcha_distance(bg_path, slider_path)
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
                    logger.info(f"第 {attempt + 1} 次滑动完成！等待系统校验...")
                    await asyncio.sleep(3)
                
                if await slider_container.is_visible():
                    logger.warning("3次滑块尝试均未通过！系统可能风控较严。")
            
            except Exception as e:
                logger.info(f"未检测到滑块或无需验证: {e}")
            
            await page.wait_for_load_state("networkidle")
            
            # 3. 进入个人空间
            logger.info("正在点击个人空间...")
            async with context.expect_page() as center_page_info:
                await page.locator('text="个人空间"').click()
            center_page = await center_page_info.value
            await center_page.wait_for_load_state("networkidle")
            
            # 4. 点击对应课程
            logger.info(f"正在寻找课程卡片: {course_name}...")
            course_link = None
            direct_locator = center_page.locator(
                f'.myde_course_item[cname="{course_name}"] a'
            ).first
            frame_locator = center_page.frame_locator('iframe').first.locator(
                f'.myde_course_item[cname="{course_name}"] a'
            ).first
            
            try:
                await direct_locator.wait_for(state="visible", timeout=3000)
                course_link = direct_locator
                logger.info("成功：在主页面找到了课程！")
            except Exception:
                logger.info("主页面未找到，正在穿透 iframe 寻找...")
                try:
                    await frame_locator.wait_for(state="visible", timeout=15000)
                    course_link = frame_locator
                    logger.info("成功：在 iframe 中找到了课程！")
                except Exception as e:
                    error_msg = f"彻底找不到课程卡片 '{course_name}': {e}"
                    logger.error(error_msg)
                    await browser.close()
                    return error_msg
            
            logger.info("正在点击课程，等待课程主页弹窗...")
            async with context.expect_page() as course_page_info:
                await course_link.click()
            course_page = await course_page_info.value
            await course_page.wait_for_load_state("networkidle")
            await asyncio.sleep(2)
            
            # 5. 点击"作业"菜单
            logger.info("正在点击左侧导航栏的\"作业\"菜单...")
            await course_page.locator('a[title="作业"]').first.click()
            logger.info("等待作业列表加载...")
            await asyncio.sleep(3)
            
            # 6. 点击指定作业的"批阅"按钮
            logger.info(f"正在寻找作业：【{assignment_name}】 的批阅按钮...")
            
            target_piyue = None
            exact_locator_str = f'li:has(h2.list_li_tit:has-text("{assignment_name}")) a.piyueBtn:has-text("批阅")'
            
            piyue_direct = course_page.locator(exact_locator_str).first
            piyue_frame = course_page.frame_locator('iframe').first.locator(exact_locator_str).first
            
            try:
                await piyue_direct.wait_for(state="visible", timeout=3000)
                target_piyue = piyue_direct
                logger.info(f"成功：在主页面找到了【{assignment_name}】的批阅按钮！")
            except Exception:
                logger.info("主页面未找到，正在穿透 iframe 寻找...")
                try:
                    await piyue_frame.wait_for(state="visible", timeout=10000)
                    target_piyue = piyue_frame
                    logger.info(f"成功：在 iframe 中找到了【{assignment_name}】的批阅按钮！")
                except Exception as e:
                    error_msg = f"彻底找不到名为【{assignment_name}】的批阅按钮: {e}"
                    logger.error(error_msg)
                    await browser.close()
                    return error_msg
            
            logger.info("正在点击批阅按钮，等待页面跳转到批阅大厅...")
            await target_piyue.evaluate("node => node.click()")
            
            await course_page.wait_for_load_state("networkidle")
            grading_page = course_page
            await asyncio.sleep(2)
            
            # 7. 导出作业附件
            logger.info("正在寻找导出选项...")
            export_selector = 'ul.morePop a:has-text("导出作业附件")'
            working_area = None
            
            try:
                await grading_page.locator(export_selector).first.wait_for(
                    state="attached", timeout=3000
                )
                working_area = grading_page
                logger.info("成功：在主页面锁定了导出工作区！")
            except Exception:
                logger.info("主页面未找到，正在穿透 iframe 寻找...")
                working_area = grading_page.frame_locator('iframe').first
                await working_area.locator(export_selector).first.wait_for(
                    state="attached", timeout=10000
                )
                logger.info("成功：在 iframe 中锁定了导出工作区！")
            
            logger.info("绕过 Playwright 所有的可见性检查，执行底层 JS 原生点击...")
            await working_area.locator(export_selector).first.evaluate(
                "node => node.click()"
            )
            
            # 7.2 等待导出选项弹窗加载
            logger.info("等待导出设置弹窗...")
            pop_div = working_area.locator('.popDiv.centerPop').first
            await pop_div.wait_for(state="visible", timeout=5000)
            
            # 7.3 选择班级范围
            logger.info("正在选择范围: 导出作业下所有班级作业附件...")
            all_class_span = pop_div.locator('.export-range.all span.grade_check').first
            await all_class_span.evaluate("node => node.click()")
            await asyncio.sleep(0.5)
            
            # 7.4 选择导出格式
            logger.info("切换导出格式为：导出提交附件...")
            attachment_span = pop_div.locator(
                'div.out:has-text("导出提交附件") span.grade_check'
            ).first
            await attachment_span.evaluate("node => node.click()")
            await asyncio.sleep(0.5)
            
            # 7.5 点击确认导出
            logger.info("提交导出任务...")
            confirm_btn = pop_div.locator('a.confirmDown:has-text("确定")').first
            await confirm_btn.evaluate("node => node.click()")
            
            logger.info("导出指令已发送，等待系统处理...")
            await asyncio.sleep(3)
            
            # 8. 智能批量处理下载中心面板
            logger.info("检查下载中心...")
            download_center = grading_page.locator('#downloadcenter, .downloadCenter').first
            
            try:
                await download_center.wait_for(state="visible", timeout=3000)
                logger.info("下载面板已自动弹出！")
            except Exception:
                logger.info("下载面板未自动弹出，正在使用强力点击呼出...")
                download_center_btn = grading_page.locator('text="下载中心"').first
                await download_center_btn.evaluate("node => node.click()")
                await download_center.wait_for(state="visible", timeout=10000)
            
            logger.info("正在扫描并锁定新生成的打包任务...")
            rows_locator = download_center.locator('.dataBody_td, tbody tr')
            target_indices = []
            
            # 防网络延迟的轮询机制，最多尝试收集5次（10秒）
            for attempt in range(5):
                await asyncio.sleep(2)
                row_count = await rows_locator.count()
                target_indices = []
                
                for i in range(row_count):
                    row = rows_locator.nth(i)
                    text = await row.inner_text()
                    
                    # 只要状态是"导出中"或"等待"，就是新任务
                    if "导出中" in text or "等待" in text:
                        target_indices.append(i)
                    elif "导出成功" in text:
                        # 碰到了导出成功的记录
                        if len(target_indices) > 0:
                            break
                
                # 如果收集到了任务，就不需要再重试了
                if len(target_indices) > 0:
                    break
                else:
                    logger.info(
                        f"第 {attempt + 1} 次扫描未发现'导出中'的任务，"
                        "可能列表正准备刷新，稍后重试..."
                    )
            
            if not target_indices:
                logger.warning(
                    "未能捕获到任何处于'导出中'的任务！"
                    "可能是系统秒速打包完毕，或网络极度卡顿。"
                )
            else:
                logger.info(f"成功锁定前 {len(target_indices)} 个正在排队的新任务！")
                logger.info("正在持续监控打包状态 (最长等待120秒)，请耐心等待...")
                
                timeout = 120
                start_time = time.time()
                
                # 轮询等待监控池里的任务全部变成"导出成功"
                while True:
                    if time.time() - start_time > timeout:
                        logger.warning("等待超时，部分班级可能未能完成打包！")
                        break
                    
                    all_success = True
                    for i in target_indices:
                        row = rows_locator.nth(i)
                        text = await row.inner_text()
                        
                        if "导出中" in text or "等待" in text:
                            all_success = False
                            break
                    
                    if all_success:
                        logger.info("🎉 监控池内所有班级均已打包完成！")
                        break
                    
                    await asyncio.sleep(3)
                
                logger.info("准备批量下载...")
                
                # 依次对我们收集的行进行下载
                for i in target_indices:
                    row = rows_locator.nth(i)
                    download_btn = row.locator('a.download_ic, a:has-text("下载")').first
                    
                    logger.info(f"正在拦截第 {i+1} 个文件的真实下载流...")
                    try:
                        async with grading_page.expect_download(timeout=60000) as download_info:
                            await download_btn.evaluate("node => node.click()")
                        
                        download = await download_info.value
                        original_name = download.suggested_filename
                        save_path = os.path.join(save_dir, original_name)
                        
                        await download.save_as(save_path)
                        download_paths.append(save_path)
                        logger.info(f"✅ 第 {i+1} 个文件下载成功: {save_path}")
                    except Exception as e:
                        logger.error(f"第 {i+1} 个文件下载失败: {e}")
                    
                    # 下载完一个等一等，防止卡崩
                    await asyncio.sleep(1.5)
                
                logger.info(f"批量下载大获全胜！共成功下载 {len(download_paths)} 个压缩包。")
            
            logger.info("任务执行完毕，清理资源中...")
            await browser.close()
            
            if download_paths:
                logger.info(f"RPA任务成功完成，返回 {len(download_paths)} 个文件路径")
                return download_paths
            else:
                error_msg = "未能下载任何文件，请检查课程名称和作业名称是否正确"
                logger.error(error_msg)
                return error_msg
    
    except Exception as e:
        error_msg = f"RPA抓取过程中发生错误: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return error_msg


# LangChain Tool装饰器
try:
    from langchain_core.tools import tool
    
    @tool
    def fetch_homework_tool(
        username: str,
        password: str,
        course_name: str,
        assignment_name: str,
        save_dir: str = DEFAULT_SAVE_DIR
    ) -> Union[List[str], str]:
        """
        从教务系统下载作业附件。当用户要求从教务系统下载并批改作业时调用此工具。
        
        Args:
            username: 教务系统用户名
            password: 教务系统密码
            course_name: 课程名称（需完全匹配）
            assignment_name: 作业名称
            save_dir: 文件保存目录，默认为/tmp/ai_grade_homeworks
        
        Returns:
            成功时返回ZIP文件绝对路径列表，失败时返回错误信息字符串
        """
        return asyncio.run(fetch_homework_from_portal(
            username, password, course_name, assignment_name, save_dir
        ))
    
    logger.info("LangChain Tool 'fetch_homework_tool' 注册成功")
    
except ImportError:
    logger.warning("LangChain未安装，跳过Tool装饰器，仅保留异步函数")
    fetch_homework_tool = None


# 便捷的同步调用函数
def fetch_homework_sync(
    username: str,
    password: str,
    course_name: str,
    assignment_name: str,
    save_dir: str = DEFAULT_SAVE_DIR
) -> Union[List[str], str]:
    """
    同步版本的作业下载函数（供gRPC Server调用）
    
    Args:
        username: 教务系统用户名
        password: 教务系统密码
        course_name: 课程名称
        assignment_name: 作业名称
        save_dir: 保存目录
    
    Returns:
        文件路径列表或错误信息
    """
    return asyncio.run(fetch_homework_from_portal(
        username, password, course_name, assignment_name, save_dir
    ))