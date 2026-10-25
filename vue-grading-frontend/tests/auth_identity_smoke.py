"""Authentication UI identity regression with fixture APIs only; no real accounts."""
import asyncio
import json
import os
import re
import urllib.parse
from playwright.async_api import async_playwright, expect


async def main():
    base=os.environ.get('RPA_UI_BASE_URL','http://127.0.0.1:6006')
    errors=[]
    async with async_playwright() as pw:
        browser=await pw.chromium.launch(headless=True,args=['--no-sandbox'])
        context=await browser.new_context()
        page=await context.new_page();page.on('pageerror',lambda error:errors.append(str(error)))
        posts=[];ready=asyncio.Event();release=asyncio.Event()
        async def register_api(route):
            req=route.request;path=urllib.parse.urlparse(req.url).path
            if path=='/api/sessions':data=[{'id':'fixture-session','title':'Fixture'}] if req.method=='GET' else {'id':'fixture-session'}
            elif path.endswith('/history'):data={'messages':[]}
            elif req.method=='POST':
                assert path=='/api/register'
                posts.append(req.post_data_json);ready.set();await release.wait()
                data=dict(user_id=99,username='canonical-user',name='正确姓名',role='teacher',access_token='fixture-access',refresh_token='fixture-refresh')
            elif path=='/api/profile':data=dict(id=99,username='canonical-user',name='正确姓名',role='teacher')
            else:data=[]
            await route.fulfill(content_type='application/json',body=json.dumps(data,ensure_ascii=False))
        await page.route('**/api/**',register_api)
        await page.route('**/assignments/**',register_api)
        await page.goto(base+'/login')
        await expect(page.get_by_label('教师账号',exact=True)).to_be_visible()
        await expect(page.get_by_label('登录密码',exact=True)).to_be_visible()
        await expect(page.locator('#username')).to_have_attribute('autocomplete','section-login username')
        await expect(page.locator('#password')).to_have_attribute('autocomplete','section-login current-password')
        await page.get_by_role('button',name='注册新账户',exact=True).click()
        await expect(page.get_by_label('真实姓名',exact=True)).to_be_visible()
        await expect(page.locator('#reg-password')).to_have_attribute('autocomplete','section-register new-password')
        await expect(page.locator('#reg-name')).to_have_attribute('autocomplete','section-register name')
        await page.locator('#reg-username').fill('typed-user')
        await page.locator('#reg-password').fill('fixture-secret')
        await page.locator('#reg-name').fill('fixture-secret')
        assert await page.locator('#reg-role').count()==0
        await page.get_by_role('button',name='注册',exact=True).click()
        await expect(page.get_by_role('alert')).to_contain_text('姓名不能与密码相同')
        assert not posts,'Unsafe name must be blocked before request'
        # Password managers can update DOM values without Vue input events.
        await page.evaluate('''() => {
          document.querySelector('#reg-username').value='autofilled-user';
          document.querySelector('#reg-password').value='autofilled-fixture-secret';
          document.querySelector('#reg-name').value='自动填写的姓名';
          document.querySelector('form[name=register]').requestSubmit();
        }''')
        await asyncio.wait_for(ready.wait(),timeout=10)
        assert posts==[dict(username='autofilled-user',password='autofilled-fixture-secret',name='自动填写的姓名',role='teacher')]
        await expect(page.locator('#reg-username')).to_be_disabled()
        await expect(page.get_by_role('button',name='返回登录',exact=True)).to_be_disabled()
        release.set()
        await expect(page).to_have_url(re.compile('/assignments$'))
        await expect(page.locator('.sidebar-user .user-info p')).to_have_text('正确姓名')
        await expect(page.locator('.sidebar-user .user-info p')).to_have_attribute('title','真实姓名')
        await expect(page.locator('.sidebar-user .user-avatar')).to_have_text('正')
        info=await page.evaluate('JSON.parse(localStorage.getItem("user_info"))')
        assert info['username']=='canonical-user' and info['name']=='正确姓名'
        assert 'autofilled-fixture-secret' not in json.dumps(info)
        # A stale cache containing the former bad display name must not expose it in the sidebar.
        await page.evaluate('''() => {const u=JSON.parse(localStorage.getItem('user_info'));u.name='stale-fixture-secret';localStorage.setItem('user_info',JSON.stringify(u))}''')
        await page.reload()
        await expect(page.locator('.sidebar-user .user-info p')).to_have_text('正确姓名')
        assert 'stale-fixture-secret' not in await page.locator('.sidebar-user').inner_text()
        await page.wait_for_function("JSON.parse(localStorage.getItem('user_info')).name === '正确姓名'")
        await context.close()
        for role in ['teacher','admin']:
            context=await browser.new_context();page=await context.new_page()
            page.on('pageerror',lambda error:errors.append(str(error)))
            await context.add_init_script('localStorage.setItem("current_session_id","fixture-session")')
            requests=[]
            async def login_api(route):
                req=route.request;path=urllib.parse.urlparse(req.url).path
                if path=='/api/sessions':data=[{'id':'fixture-session','title':'Fixture'}] if req.method=='GET' else {'id':'fixture-session'}
                elif req.method=='POST':
                    assert path=='/api/login';requests.append(req.post_data_json)
                    data=dict(user_id=99,username='canonical-'+role,name='独立姓名',role=role,access_token='fixture-access',refresh_token='fixture-refresh')
                elif path.endswith('/history'):data=dict(messages=[])
                elif path=='/api/profile':data=dict(id=99,username='canonical-'+role,name='独立姓名',role=role)
                else:data=[]
                await route.fulfill(content_type='application/json',body=json.dumps(data,ensure_ascii=False))
            await page.route('**/api/**',login_api)
            await page.route('**/assignments/**',login_api)
            await page.goto(base+'/login')
            await page.locator('#username').fill('entered-user')
            await page.locator('#password').fill('login-fixture-secret')
            await page.get_by_role('button',name='登录',exact=True).click()
            await expect(page).to_have_url(re.compile('/assignments$'))
            await expect(page.locator('.sidebar-user .user-info p')).to_have_text('独立姓名')
            assert (await page.evaluate('JSON.parse(localStorage.getItem("user_info"))'))['username']=='canonical-'+role
            assert requests==[dict(username='entered-user',password='login-fixture-secret')]
            await context.close()
        await browser.close()
    assert not errors,errors
    print('PASS: distinct account/password/real-name labels, real-name display and avatar, canonical login identity, cache refresh, autofill handling and role redirects; fixture APIs only.')


asyncio.run(main())
