"use strict";(self.webpackChunkflowtorch=self.webpackChunkflowtorch||[]).push([[4489],{6705:function(e,s,o){o.r(s),o.d(s,{frontMatter:function(){return p},contentTitle:function(){return h},metadata:function(){return b},toc:function(){return k},default:function(){return N}});var a=o(7462),t=o(3366),n=(o(7294),o(3905)),l=o(2814),c=o(1436),r=o(1032),i=(o(8666),o(2520)),m=(o(84),o(7868)),d=["components"],p={id:"flowtorch.bijectors.spline",sidebar_label:"Spline"},h=void 0,b={unversionedId:"api/flowtorch.bijectors.spline",id:"api/flowtorch.bijectors.spline",isDocsHomePage:!1,title:"flowtorch.bijectors.spline",description:"flowtorch  bijectors  Spline",source:"@site/docs/api/flowtorch.bijectors.spline.mdx",sourceDirName:"api",slug:"/api/flowtorch.bijectors.spline",permalink:"/api/flowtorch.bijectors.spline",editUrl:"https://github.com/facebookincubator/flowtorch/edit/main/website/docs/api/flowtorch.bijectors.spline.mdx",tags:[],version:"current",frontMatter:{id:"flowtorch.bijectors.spline",sidebar_label:"Spline"},sidebar:"apiSidebar",previous:{title:"Softplus",permalink:"/api/flowtorch.bijectors.softplus"},next:{title:"SplineAutoregressive",permalink:"/api/flowtorch.bijectors.splineautoregressive"}},k=[{value:'<span className="doc-symbol-name">flowtorch.bijectors.Spline</span>',id:"class",children:[{value:'<span className="doc-symbol-name">__init__</span>',id:"--init--",children:[],level:3},{value:'<span className="doc-symbol-name">forward</span>',id:"forward",children:[],level:3},{value:'<span className="doc-symbol-name">forward_shape</span>',id:"forward-shape",children:[],level:3},{value:'<span className="doc-symbol-name">inverse</span>',id:"inverse",children:[],level:3},{value:'<span className="doc-symbol-name">inverse_shape</span>',id:"inverse-shape",children:[],level:3},{value:'<span className="doc-symbol-name">log_abs_det_jacobian</span>',id:"log-abs-det-jacobian",children:[],level:3},{value:'<span className="doc-symbol-name">param_shapes</span>',id:"param-shapes",children:[],level:3}],level:2}],u={toc:k};function N(e){var s=e.components,o=(0,t.Z)(e,d);return(0,n.kt)("wrapper",(0,a.Z)({},u,o,{components:s,mdxType:"MDXLayout"}),(0,n.kt)(m.Z,{url:"https://github.com/facebookincubator/flowtorch/blob/main/flowtorch/bijectors/spline.py",mdxType:"PythonNavbar"},(0,n.kt)("p",null,(0,n.kt)("a",{parentName:"p",href:"/api/flowtorch"},"flowtorch")," ",(0,n.kt)(l.G,{icon:c.cLY,size:"sm",mdxType:"FontAwesomeIcon"})," ",(0,n.kt)("a",{parentName:"p",href:"/api/flowtorch.bijectors"},"bijectors")," ",(0,n.kt)(l.G,{icon:c.cLY,size:"sm",mdxType:"FontAwesomeIcon"})," ",(0,n.kt)("em",{parentName:"p"},"Spline"))),(0,n.kt)(r.Z,{mdxType:"PythonClass"},(0,n.kt)("div",{className:"doc-class-row"},(0,n.kt)("div",{className:"doc-class-label"},(0,n.kt)("span",{className:"doc-symbol-label"},"class")),(0,n.kt)("div",{className:"doc-class-signature"},(0,n.kt)("h2",{id:"class"},(0,n.kt)("span",{className:"doc-symbol-name"},"flowtorch.bijectors.Spline")),(0,n.kt)("span",{className:"doc-inherits-from"},"Inherits from: ",(0,n.kt)("span",{className:"doc-symbol-name"},"flowtorch.bijectors.ops.spline.Spline, flowtorch.bijectors.elementwise.Elementwise"))))),(0,n.kt)("pre",null,(0,n.kt)("code",{parentName:"pre"},"empty docstring\n")),(0,n.kt)(i.Z,{mdxType:"PythonMethod"},(0,n.kt)("div",{className:"doc-method-row"},(0,n.kt)("div",{className:"doc-method-label"},(0,n.kt)("span",{className:"doc-symbol-label"},"member")),(0,n.kt)("div",{className:"doc-method-signature"},(0,n.kt)("h3",{id:"--init--"},(0,n.kt)("span",{className:"doc-symbol-name"},"_","_","init","_","_")),(0,n.kt)("span",{className:"doc-symbol-signature"},"(self, params: Union[flowtorch.lazy.Lazy, NoneType] = None, *, shape: torch.Size, context_shape: Union[torch.Size, NoneType] = None, count_bins: int = 8, bound: float = 3.0, order: str = 'linear') -> None")))),(0,n.kt)("pre",null,(0,n.kt)("code",{parentName:"pre"},"<empty docstring>\n")),(0,n.kt)(i.Z,{mdxType:"PythonMethod"},(0,n.kt)("div",{className:"doc-method-row"},(0,n.kt)("div",{className:"doc-method-label"},(0,n.kt)("span",{className:"doc-symbol-label"},"member")),(0,n.kt)("div",{className:"doc-method-signature"},(0,n.kt)("h3",{id:"forward"},(0,n.kt)("span",{className:"doc-symbol-name"},"forward")),(0,n.kt)("span",{className:"doc-symbol-signature"},"(self, x: torch.Tensor, context: Union[torch.Tensor, NoneType] = None) -> torch.Tensor")))),(0,n.kt)("pre",null,(0,n.kt)("code",{parentName:"pre"},"<empty docstring>\n")),(0,n.kt)(i.Z,{mdxType:"PythonMethod"},(0,n.kt)("div",{className:"doc-method-row"},(0,n.kt)("div",{className:"doc-method-label"},(0,n.kt)("span",{className:"doc-symbol-label"},"member")),(0,n.kt)("div",{className:"doc-method-signature"},(0,n.kt)("h3",{id:"forward-shape"},(0,n.kt)("span",{className:"doc-symbol-name"},"forward","_","shape")),(0,n.kt)("span",{className:"doc-symbol-signature"},"(self, shape: torch.Size) -> torch.Size")))),(0,n.kt)("pre",null,(0,n.kt)("code",{parentName:"pre"},"\nInfers the shape of the forward computation, given the input shape.\nDefaults to preserving shape.\n\n")),(0,n.kt)(i.Z,{mdxType:"PythonMethod"},(0,n.kt)("div",{className:"doc-method-row"},(0,n.kt)("div",{className:"doc-method-label"},(0,n.kt)("span",{className:"doc-symbol-label"},"member")),(0,n.kt)("div",{className:"doc-method-signature"},(0,n.kt)("h3",{id:"inverse"},(0,n.kt)("span",{className:"doc-symbol-name"},"inverse")),(0,n.kt)("span",{className:"doc-symbol-signature"},"(self, y: torch.Tensor, x: Union[torch.Tensor, NoneType] = None, context: Union[torch.Tensor, NoneType] = None) -> torch.Tensor")))),(0,n.kt)("pre",null,(0,n.kt)("code",{parentName:"pre"},"<empty docstring>\n")),(0,n.kt)(i.Z,{mdxType:"PythonMethod"},(0,n.kt)("div",{className:"doc-method-row"},(0,n.kt)("div",{className:"doc-method-label"},(0,n.kt)("span",{className:"doc-symbol-label"},"member")),(0,n.kt)("div",{className:"doc-method-signature"},(0,n.kt)("h3",{id:"inverse-shape"},(0,n.kt)("span",{className:"doc-symbol-name"},"inverse","_","shape")),(0,n.kt)("span",{className:"doc-symbol-signature"},"(self, shape: torch.Size) -> torch.Size")))),(0,n.kt)("pre",null,(0,n.kt)("code",{parentName:"pre"},"\nInfers the shapes of the inverse computation, given the output shape.\nDefaults to preserving shape.\n\n")),(0,n.kt)(i.Z,{mdxType:"PythonMethod"},(0,n.kt)("div",{className:"doc-method-row"},(0,n.kt)("div",{className:"doc-method-label"},(0,n.kt)("span",{className:"doc-symbol-label"},"member")),(0,n.kt)("div",{className:"doc-method-signature"},(0,n.kt)("h3",{id:"log-abs-det-jacobian"},(0,n.kt)("span",{className:"doc-symbol-name"},"log","_","abs","_","det","_","jacobian")),(0,n.kt)("span",{className:"doc-symbol-signature"},"(self, x: torch.Tensor, y: torch.Tensor, context: Union[torch.Tensor, NoneType] = None) -> torch.Tensor")))),(0,n.kt)("pre",null,(0,n.kt)("code",{parentName:"pre"},"\nComputes the log det jacobian `log |dy/dx|` given input and output.\nBy default, assumes a volume preserving bijection.\n\n")),(0,n.kt)(i.Z,{mdxType:"PythonMethod"},(0,n.kt)("div",{className:"doc-method-row"},(0,n.kt)("div",{className:"doc-method-label"},(0,n.kt)("span",{className:"doc-symbol-label"},"member")),(0,n.kt)("div",{className:"doc-method-signature"},(0,n.kt)("h3",{id:"param-shapes"},(0,n.kt)("span",{className:"doc-symbol-name"},"param","_","shapes")),(0,n.kt)("span",{className:"doc-symbol-signature"},"(self, shape: torch.Size) -> Sequence[torch.Size]")))),(0,n.kt)("pre",null,(0,n.kt)("code",{parentName:"pre"},"<empty docstring>\n")))}N.isMDXComponent=!0}}]);