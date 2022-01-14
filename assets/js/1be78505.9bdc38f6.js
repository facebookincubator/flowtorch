"use strict";(self.webpackChunkflowtorch=self.webpackChunkflowtorch||[]).push([[9514,4608],{4608:function(e,t,a){a.r(t);var n=a(7294),r=a(1022),o=a(5999);t.default=function(){return n.createElement(r.Z,{title:(0,o.I)({id:"theme.NotFound.title",message:"Page Not Found"})},n.createElement("main",{className:"container margin-vert--xl"},n.createElement("div",{className:"row"},n.createElement("div",{className:"col col--6 col--offset-3"},n.createElement("h1",{className:"hero__title"},n.createElement(o.Z,{id:"theme.NotFound.title",description:"The title of the 404 page"},"Page Not Found")),n.createElement("p",null,n.createElement(o.Z,{id:"theme.NotFound.p1",description:"The first paragraph of the 404 page"},"We could not find what you were looking for.")),n.createElement("p",null,n.createElement(o.Z,{id:"theme.NotFound.p2",description:"The 2nd paragraph of the 404 page"},"Please contact the owner of the site that linked you to the original URL and let them know their link is broken."))))))}},8341:function(e,t,a){a.r(t),a.d(t,{default:function(){return te}});var n=a(7294),r=a(3905),o=a(6291),l=a(1022),c=a(6010),i=a(9521),s=a(3783),d=a(5537),m=a(7462),u=function(e){return n.createElement("svg",(0,m.Z)({width:"20",height:"20","aria-hidden":"true"},e),n.createElement("g",{fill:"#7a7a7a"},n.createElement("path",{d:"M9.992 10.023c0 .2-.062.399-.172.547l-4.996 7.492a.982.982 0 01-.828.454H1c-.55 0-1-.453-1-1 0-.2.059-.403.168-.551l4.629-6.942L.168 3.078A.939.939 0 010 2.528c0-.548.45-.997 1-.997h2.996c.352 0 .649.18.828.45L9.82 9.472c.11.148.172.347.172.55zm0 0"}),n.createElement("path",{d:"M19.98 10.023c0 .2-.058.399-.168.547l-4.996 7.492a.987.987 0 01-.828.454h-3c-.547 0-.996-.453-.996-1 0-.2.059-.403.168-.551l4.625-6.942-4.625-6.945a.939.939 0 01-.168-.55 1 1 0 01.996-.997h3c.348 0 .649.18.828.45l4.996 7.492c.11.148.168.347.168.55zm0 0"})))},p=a(5999),b=a(3366),h=a(9960),f=a(3919),v=a(541),E="menuLinkText_OKON",k=["items"],g=["item"],C=["item","onItemClick","activePath","level"],_=["item","onItemClick","activePath","level"],Z=function e(t,a){return"link"===t.type?(0,i.Mg)(t.href,a):"category"===t.type&&t.items.some((function(t){return e(t,a)}))},S=(0,n.memo)((function(e){var t=e.items,a=(0,b.Z)(e,k);return n.createElement(n.Fragment,null,t.map((function(e,t){return n.createElement(N,(0,m.Z)({key:t,item:e},a))})))}));function N(e){var t=e.item,a=(0,b.Z)(e,g);return"category"===t.type?0===t.items.length?null:n.createElement(I,(0,m.Z)({item:t},a)):n.createElement(T,(0,m.Z)({item:t},a))}function I(e){var t,a=e.item,r=e.onItemClick,o=e.activePath,l=e.level,s=(0,b.Z)(e,C),d=a.items,u=a.label,p=a.collapsible,h=a.className,f=Z(a,o),v=(0,i.uR)({initialState:function(){return!!p&&(!f&&a.collapsed)}}),k=v.collapsed,g=v.setCollapsed,_=v.toggleCollapsed;return function(e){var t=e.isActive,a=e.collapsed,r=e.setCollapsed,o=(0,i.D9)(t);(0,n.useEffect)((function(){t&&!o&&a&&r(!1)}),[t,o,a,r])}({isActive:f,collapsed:k,setCollapsed:g}),n.createElement("li",{className:(0,c.Z)(i.kM.docs.docSidebarItemCategory,i.kM.docs.docSidebarItemCategoryLevel(l),"menu__list-item",{"menu__list-item--collapsed":k},h)},n.createElement("a",(0,m.Z)({className:(0,c.Z)("menu__link",(t={"menu__link--sublist":p,"menu__link--active":p&&f},t[E]=!p,t)),onClick:p?function(e){e.preventDefault(),_()}:void 0,href:p?"#":void 0},s),u),n.createElement(i.zF,{lazy:!0,as:"ul",className:"menu__list",collapsed:k},n.createElement(S,{items:d,tabIndex:k?-1:0,onItemClick:r,activePath:o,level:l+1})))}function T(e){var t=e.item,a=e.onItemClick,r=e.activePath,o=e.level,l=(0,b.Z)(e,_),s=t.href,d=t.label,u=t.className,p=Z(t,r);return n.createElement("li",{className:(0,c.Z)(i.kM.docs.docSidebarItemLink,i.kM.docs.docSidebarItemLinkLevel(o),"menu__list-item",u),key:d},n.createElement(h.Z,(0,m.Z)({className:(0,c.Z)("menu__link",{"menu__link--active":p}),"aria-current":p?"page":void 0,to:s},(0,f.Z)(s)&&{onClick:a},l),(0,f.Z)(s)?d:n.createElement("span",null,d,n.createElement(v.Z,null))))}var M="sidebar_a3j0",w="sidebarWithHideableNavbar_VlPv",y="sidebarHidden_OqfG",F="sidebarLogo_hmkv",A="menu_cyFh",L="menuWithAnnouncementBar_+O1J",B="collapseSidebarButton_eoK2",P="collapseSidebarButtonIcon_e+kA";function x(e){var t=e.onClick;return n.createElement("button",{type:"button",title:(0,p.I)({id:"theme.docs.sidebar.collapseButtonTitle",message:"Collapse sidebar",description:"The title attribute for collapse button of doc sidebar"}),"aria-label":(0,p.I)({id:"theme.docs.sidebar.collapseButtonAriaLabel",message:"Collapse sidebar",description:"The title attribute for collapse button of doc sidebar"}),className:(0,c.Z)("button button--secondary button--outline",B),onClick:t},n.createElement(u,{className:P}))}function R(e){var t,a,r=e.path,o=e.sidebar,l=e.onCollapse,s=e.isHidden,m=function(){var e=(0,i.nT)().isActive,t=(0,n.useState)(e),a=t[0],r=t[1];return(0,i.RF)((function(t){var a=t.scrollY;e&&r(0===a)}),[e]),e&&a}(),u=(0,i.LU)(),p=u.navbar.hideOnScroll,b=u.hideableSidebar;return n.createElement("div",{className:(0,c.Z)(M,(t={},t[w]=p,t[y]=s,t))},p&&n.createElement(d.Z,{tabIndex:-1,className:F}),n.createElement("nav",{className:(0,c.Z)("menu thin-scrollbar",A,(a={},a[L]=m,a))},n.createElement("ul",{className:(0,c.Z)(i.kM.docs.docSidebarMenu,"menu__list")},n.createElement(S,{items:o,activePath:r,level:1}))),b&&n.createElement(x,{onClick:l}))}var H=function(e){var t=e.toggleSidebar,a=e.sidebar,r=e.path;return n.createElement("ul",{className:(0,c.Z)(i.kM.docs.docSidebarMenu,"menu__list")},n.createElement(S,{items:a,activePath:r,onItemClick:function(){return t()},level:1}))};function D(e){return n.createElement(i.Cv,{component:H,props:e})}var W=n.memo(R),O=n.memo(D);function z(e){var t=(0,s.Z)(),a="desktop"===t||"ssr"===t,r="mobile"===t;return n.createElement(n.Fragment,null,a&&n.createElement(W,e),r&&n.createElement(O,e))}function Y(e){return n.createElement(n.Fragment,null,n.createElement("div",{id:e.path},n.createElement(z,e)))}var q=a(9028),K=a(4608),U="backToTopButton_i9tI",X="backToTopButtonShow_wCmF";function j(){var e=(0,n.useRef)(null);return{smoothScrollTop:function(){var t;e.current=(t=null,function e(){var a=document.documentElement.scrollTop;a>0&&(t=requestAnimationFrame(e),window.scrollTo(0,Math.floor(.85*a)))}(),function(){return t&&cancelAnimationFrame(t)})},cancelScrollToTop:function(){return null==e.current?void 0:e.current()}}}var G=function(){var e,t=(0,n.useState)(!1),a=t[0],r=t[1],o=(0,n.useRef)(!1),l=j(),s=l.smoothScrollTop,d=l.cancelScrollToTop;return(0,i.RF)((function(e,t){var a=e.scrollY,n=null==t?void 0:t.scrollY;if(n)if(o.current)o.current=!1;else{var l=a<n;if(l||d(),a<300)r(!1);else if(l){var c=document.documentElement.scrollHeight;a+window.innerHeight<c&&r(!0)}else r(!1)}})),(0,i.SL)((function(e){e.location.hash&&(o.current=!0,r(!1))})),n.createElement("button",{"aria-label":(0,p.I)({id:"theme.BackToTopButton.buttonAriaLabel",message:"Scroll back to top",description:"The ARIA label for the back to top button"}),className:(0,c.Z)("clean-btn",i.kM.common.backToTopButton,U,(e={},e[X]=a,e)),type:"button",onClick:function(){return s()}})},J=a(6775),Q={docPage:"docPage_lDyR",docMainContainer:"docMainContainer_r8cw",docSidebarContainer:"docSidebarContainer_0YBq",docMainContainerEnhanced:"docMainContainerEnhanced_SOUu",docSidebarContainerHidden:"docSidebarContainerHidden_Qlt2",collapsedDocSidebar:"collapsedDocSidebar_zZpm",expandSidebarButtonIcon:"expandSidebarButtonIcon_cxi8",docItemWrapperEnhanced:"docItemWrapperEnhanced_aT5H"},V=a(2859);function $(e){var t,a,o,s=e.currentDocRoute,d=e.versionMetadata,m=e.children,b=d.pluginId,h=d.version,f=s.sidebar,v=f?d.docsSidebars[f]:void 0,E=(0,n.useState)(!1),k=E[0],g=E[1],C=(0,n.useState)(!1),_=C[0],Z=C[1],S=(0,n.useCallback)((function(){_&&Z(!1),g((function(e){return!e}))}),[_]);return n.createElement(l.Z,{wrapperClassName:i.kM.wrapper.docsPages,pageClassName:i.kM.page.docsDocPage,searchMetadatas:{version:h,tag:(0,i.os)(b,h)}},n.createElement("div",{className:Q.docPage},n.createElement(G,null),v&&n.createElement("aside",{className:(0,c.Z)(Q.docSidebarContainer,(t={},t[Q.docSidebarContainerHidden]=k,t)),onTransitionEnd:function(e){e.currentTarget.classList.contains(Q.docSidebarContainer)&&k&&Z(!0)}},n.createElement(Y,{key:f,sidebar:v,path:s.path,onCollapse:S,isHidden:_}),_&&n.createElement("div",{className:Q.collapsedDocSidebar,title:(0,p.I)({id:"theme.docs.sidebar.expandButtonTitle",message:"Expand sidebar",description:"The ARIA label and title attribute for expand button of doc sidebar"}),"aria-label":(0,p.I)({id:"theme.docs.sidebar.expandButtonAriaLabel",message:"Expand sidebar",description:"The ARIA label and title attribute for expand button of doc sidebar"}),tabIndex:0,role:"button",onKeyDown:S,onClick:S},n.createElement(u,{className:Q.expandSidebarButtonIcon}))),n.createElement("main",{className:(0,c.Z)(Q.docMainContainer,(a={},a[Q.docMainContainerEnhanced]=k||!v,a))},n.createElement("div",{className:(0,c.Z)("container padding-top--md padding-bottom--lg",Q.docItemWrapper,(o={},o[Q.docItemWrapperEnhanced]=k,o))},n.createElement(r.Zo,{components:q.Z},m)))))}var ee=function(e){var t=e.route.routes,a=e.versionMetadata,r=e.location,l=t.find((function(e){return(0,J.LX)(r.pathname,e)}));return l?n.createElement(n.Fragment,null,n.createElement(V.Z,null,n.createElement("html",{className:a.className})),n.createElement($,{currentDocRoute:l,versionMetadata:a},(0,o.Z)(t,{versionMetadata:a}))):n.createElement(K.default,null)};var te=function(e){var t=e.route.routes,a=(e.versionMetadata,e.location),r=t.find((function(e){return(0,J.LX)(a.pathname,e)}));return r?n.createElement(n.Fragment,null,n.createElement("div",{id:"docpage"+r.path},n.createElement(ee,e))):n.createElement(K.default,null)}}}]);