import{e as y,i as d,h as p,w as f,j as A,k as h,l as m,m as b,n as g,o as D,c as v,p as r,q as B}from"./index-4a6feb7f.js";let E;function _(){return E}function C(s){return typeof s=="function"?s():y(s)}function l(s,e=""){if(s instanceof Promise)return s;const n=C(s);return!s||!n?n:Array.isArray(n)?n.map(a=>l(a,e)):typeof n=="object"?Object.fromEntries(Object.entries(n).map(([a,o])=>a==="titleTemplate"||a.startsWith("on")?[a,y(o)]:[a,l(o,a)])):n}const k="usehead",c=typeof globalThis<"u"?globalThis:typeof window<"u"?window:typeof global<"u"?global:typeof self<"u"?self:{},i="__unhead_injection_handler__";function w(){if(i in c)return c[i]();const s=d(k);return s||_()}function q(s,e={}){const n=e.head||w();if(n)return n.ssr?n.push(s,e):x(n,s,e)}function x(s,e,n={}){const a=p(!1),o=p({});f(()=>{o.value=a.value?{}:l(e)});const t=s.push(o.value,n);return A(o,u=>{t.patch(u)}),g()&&(h(()=>{t.dispose()}),m(()=>{a.value=!0}),b(()=>{a.value=!1})),t}const I={class:"prose prose-sm m-auto text-left"},H=B(`<div class="text-center"><h3>About</h3></div><p><a href="https://github.com/antfu/aiofauna" target="_blank" rel="noopener">AioFauna</a> is an opinionated Python backend framework made by <a href="https://github.com/aiofauna" target="_blank" rel="noopener">@obahamonde</a> for building backend applications swiftly. With <strong>automatic Swagger UI docs</strong>, <strong>FastAPI-like syntax</strong>, <strong>FaunaModel ODM out of the box</strong> uses <strong>APIClient</strong> for seamless integrations.</p><pre hidden></pre><div class="shiki-container language-python"><pre class="shiki shiki-dark vitesse-dark" style="background-color:#121212;" tabindex="0"><code><span class="line"><span style="color:#4D9375;">from</span><span style="color:#DBD7CAEE;"> aiofauna </span><span style="color:#4D9375;">import</span><span style="color:#DBD7CAEE;"> APIServer</span></span>
<span class="line"></span>
<span class="line"><span style="color:#DBD7CAEE;">app </span><span style="color:#666666;">=</span><span style="color:#DBD7CAEE;"> APIServer</span><span style="color:#666666;">()</span></span>
<span class="line"></span>
<span class="line"><span style="color:#666666;">@</span><span style="color:#80A665;">app</span><span style="color:#666666;">.</span><span style="color:#80A665;">get</span><span style="color:#666666;">(</span><span style="color:#C98A7D99;">&quot;</span><span style="color:#C98A7D;">/</span><span style="color:#C98A7D99;">&quot;</span><span style="color:#666666;">)</span></span>
<span class="line"><span style="color:#CB7676;">async</span><span style="color:#DBD7CAEE;"> </span><span style="color:#CB7676;">def</span><span style="color:#DBD7CAEE;"> </span><span style="color:#80A665;">hello</span><span style="color:#666666;">():</span></span>
<span class="line"><span style="color:#DBD7CAEE;">    </span><span style="color:#4D9375;">return</span><span style="color:#DBD7CAEE;"> </span><span style="color:#666666;">{</span><span style="color:#C98A7D99;">&quot;</span><span style="color:#C98A7D;">message</span><span style="color:#C98A7D99;">&quot;</span><span style="color:#666666;">:</span><span style="color:#DBD7CAEE;"> </span><span style="color:#C98A7D99;">&quot;</span><span style="color:#C98A7D;">Hello World</span><span style="color:#C98A7D99;">&quot;</span><span style="color:#666666;">}</span></span>
<span class="line"></span></code></pre><pre class="shiki shiki-light vitesse-light" style="background-color:#ffffff;" tabindex="0"><code><span class="line"><span style="color:#1E754F;">from</span><span style="color:#393A34;"> aiofauna </span><span style="color:#1E754F;">import</span><span style="color:#393A34;"> APIServer</span></span>
<span class="line"></span>
<span class="line"><span style="color:#393A34;">app </span><span style="color:#999999;">=</span><span style="color:#393A34;"> APIServer</span><span style="color:#999999;">()</span></span>
<span class="line"></span>
<span class="line"><span style="color:#999999;">@</span><span style="color:#59873A;">app</span><span style="color:#999999;">.</span><span style="color:#59873A;">get</span><span style="color:#999999;">(</span><span style="color:#B5695999;">&quot;</span><span style="color:#B56959;">/</span><span style="color:#B5695999;">&quot;</span><span style="color:#999999;">)</span></span>
<span class="line"><span style="color:#AB5959;">async</span><span style="color:#393A34;"> </span><span style="color:#AB5959;">def</span><span style="color:#393A34;"> </span><span style="color:#59873A;">hello</span><span style="color:#999999;">():</span></span>
<span class="line"><span style="color:#393A34;">    </span><span style="color:#1E754F;">return</span><span style="color:#393A34;"> </span><span style="color:#999999;">{</span><span style="color:#B5695999;">&quot;</span><span style="color:#B56959;">message</span><span style="color:#B5695999;">&quot;</span><span style="color:#999999;">:</span><span style="color:#393A34;"> </span><span style="color:#B5695999;">&quot;</span><span style="color:#B56959;">Hello World</span><span style="color:#B5695999;">&quot;</span><span style="color:#999999;">}</span></span>
<span class="line"></span></code></pre></div><p>Check out the <a href="https://github.com/obahamonde/aiofauna" target="_blank" rel="noopener">GitHub repo</a> for more details.</p>`,5),P=[H],U="About",T=[{property:"og:title",content:"About"},{name:"twitter:title",content:"About"}],j={__name:"about",setup(s,{expose:e}){return e({frontmatter:{title:"About",meta:[{property:"og:title",content:"About"},{name:"twitter:title",content:"About"}]}}),q({title:"About",meta:[{property:"og:title",content:"About"},{name:"twitter:title",content:"About"}]}),(o,t)=>(D(),v("div",I,P))}};typeof r=="function"&&r(j);export{j as default,T as meta,U as title};