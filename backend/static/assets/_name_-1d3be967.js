import{d as r,u,h as i,s as d,t as p,o as m,c as _,b as t,v as g,e as a}from"./index-4a6feb7f.js";const f={class:"col center p-12 gap-4 bg-accent h-100vh"},h={"text-caption":""},v=t("p",null,null,-1),B=r({__name:"[name]",props:{name:null,modelValue:null},emits:["update:modelValue"],setup(n){const o=n,l=u(),s=i("");return d(async()=>{const e=await(await fetch(`/api/${o.name}`)).json();s.value=e.message}),p("modelValue"),(c,e)=>(m(),_("main",f,[t("div",h,g(a(s)),1),v,t("div",null,[t("button",{class:"btn-del",onClick:e[0]||(e[0]=k=>a(l).back())},"Back")])]))}});export{B as default};