---
id: flowtorch
title: flowtorch
sidebar_label: flowtorch
---

<div class="section" id="submodules">
<h2>Submodules<a class="headerlink" href="#submodules" title="Permalink to this headline">¶</a></h2>
</div>
<div class="section" id="module-flowtorch.bijector">
<span id="flowtorch-bijector-module"></span><h2>flowtorch.bijector module<a class="headerlink" href="#module-flowtorch.bijector" title="Permalink to this headline">¶</a></h2>
<dl class="py class">
<dt id="flowtorch.bijector.Bijector">
<em class="property">class </em><code class="sig-prename descclassname">flowtorch.bijector.</code><code class="sig-name descname">Bijector</code><span class="sig-paren">(</span><em class="sig-param"><span class="n">param_fn</span><span class="p">:</span> <span class="n"><a class="reference internal" href="#flowtorch.param.Params" title="flowtorch.param.Params">flowtorch.param.Params</a></span></em><span class="sig-paren">)</span><a class="headerlink" href="#flowtorch.bijector.Bijector" title="Permalink to this definition">¶</a></dt>
<dd><p>Bases: <code class="xref py py-class docutils literal notranslate"><span class="pre">object</span></code></p>
<dl class="py attribute">
<dt id="flowtorch.bijector.Bijector.autoregressive">
<code class="sig-name descname">autoregressive</code><em class="property"> = False</em><a class="headerlink" href="#flowtorch.bijector.Bijector.autoregressive" title="Permalink to this definition">¶</a></dt>
<dd></dd></dl>

<dl class="py attribute">
<dt id="flowtorch.bijector.Bijector.codomain">
<code class="sig-name descname">codomain</code><em class="property">: torch.distributions.constraints.Constraint</em><em class="property"> = Real()</em><a class="headerlink" href="#flowtorch.bijector.Bijector.codomain" title="Permalink to this definition">¶</a></dt>
<dd></dd></dl>

<dl class="py attribute">
<dt id="flowtorch.bijector.Bijector.domain">
<code class="sig-name descname">domain</code><em class="property">: torch.distributions.constraints.Constraint</em><em class="property"> = Real()</em><a class="headerlink" href="#flowtorch.bijector.Bijector.domain" title="Permalink to this definition">¶</a></dt>
<dd></dd></dl>

<dl class="py method">
<dt id="flowtorch.bijector.Bijector.forward">
<code class="sig-name descname">forward</code><span class="sig-paren">(</span><em class="sig-param"><span class="n">x</span><span class="p">:</span> <span class="n">torch.Tensor</span></em>, <em class="sig-param"><span class="n">params</span><span class="p">:</span> <span class="n">Optional<span class="p">[</span>flowtorch.ParamsModule<span class="p">]</span></span></em><span class="sig-paren">)</span> &#x2192; torch.Tensor<a class="headerlink" href="#flowtorch.bijector.Bijector.forward" title="Permalink to this definition">¶</a></dt>
<dd></dd></dl>

<dl class="py method">
<dt id="flowtorch.bijector.Bijector.forward_shape">
<code class="sig-name descname">forward_shape</code><span class="sig-paren">(</span><em class="sig-param"><span class="n">event_shape</span></em><span class="sig-paren">)</span><a class="headerlink" href="#flowtorch.bijector.Bijector.forward_shape" title="Permalink to this definition">¶</a></dt>
<dd><p>Infers the shape of the forward computation, given the input shape.
Defaults to preserving shape.</p>
</dd></dl>

<dl class="py method">
<dt id="flowtorch.bijector.Bijector.inv">
<code class="sig-name descname">inv</code><span class="sig-paren">(</span><span class="sig-paren">)</span> &#x2192; <a class="reference internal" href="#flowtorch.bijector.Bijector" title="flowtorch.bijector.Bijector">flowtorch.bijector.Bijector</a><a class="headerlink" href="#flowtorch.bijector.Bijector.inv" title="Permalink to this definition">¶</a></dt>
<dd></dd></dl>

<dl class="py method">
<dt id="flowtorch.bijector.Bijector.inverse">
<code class="sig-name descname">inverse</code><span class="sig-paren">(</span><em class="sig-param"><span class="n">y</span><span class="p">:</span> <span class="n">torch.Tensor</span></em>, <em class="sig-param"><span class="n">params</span><span class="p">:</span> <span class="n">Optional<span class="p">[</span>flowtorch.ParamsModule<span class="p">]</span></span></em><span class="sig-paren">)</span> &#x2192; torch.Tensor<a class="headerlink" href="#flowtorch.bijector.Bijector.inverse" title="Permalink to this definition">¶</a></dt>
<dd></dd></dl>

<dl class="py method">
<dt id="flowtorch.bijector.Bijector.inverse_shape">
<code class="sig-name descname">inverse_shape</code><span class="sig-paren">(</span><em class="sig-param"><span class="n">event_shape</span></em><span class="sig-paren">)</span><a class="headerlink" href="#flowtorch.bijector.Bijector.inverse_shape" title="Permalink to this definition">¶</a></dt>
<dd><p>Infers the shapes of the inverse computation, given the output shape.
Defaults to preserving shape.</p>
</dd></dl>

<dl class="py method">
<dt id="flowtorch.bijector.Bijector.log_abs_det_jacobian">
<code class="sig-name descname">log_abs_det_jacobian</code><span class="sig-paren">(</span><em class="sig-param"><span class="n">x</span><span class="p">:</span> <span class="n">torch.Tensor</span></em>, <em class="sig-param"><span class="n">y</span><span class="p">:</span> <span class="n">torch.Tensor</span></em>, <em class="sig-param"><span class="n">params</span><span class="p">:</span> <span class="n">Optional<span class="p">[</span>flowtorch.ParamsModule<span class="p">]</span></span></em><span class="sig-paren">)</span> &#x2192; torch.Tensor<a class="headerlink" href="#flowtorch.bijector.Bijector.log_abs_det_jacobian" title="Permalink to this definition">¶</a></dt>
<dd><p>Computes the log det jacobian $log |dy/dx|+\int^x_yx^2$ given input and output.
By default, assumes a volume preserving bijection.</p>
</dd></dl>

<dl class="py attribute">
<dt id="flowtorch.bijector.Bijector.near_identity_initialization">
<code class="sig-name descname">near_identity_initialization</code><em class="property"> = True</em><a class="headerlink" href="#flowtorch.bijector.Bijector.near_identity_initialization" title="Permalink to this definition">¶</a></dt>
<dd></dd></dl>

<dl class="py method">
<dt id="flowtorch.bijector.Bijector.param_shapes">
<code class="sig-name descname">param_shapes</code><span class="sig-paren">(</span><em class="sig-param"><span class="n">dist</span><span class="p">:</span> <span class="n">torch.distributions.distribution.Distribution</span></em><span class="sig-paren">)</span> &#x2192; Sequence<span class="p">[</span>torch.Size<span class="p">]</span><a class="headerlink" href="#flowtorch.bijector.Bijector.param_shapes" title="Permalink to this definition">¶</a></dt>
<dd><p>Given a base distribution, calculate the parameters for the transformation
of that distribution under this bijector. By default, no parameters are
set.</p>
</dd></dl>

<dl class="py attribute">
<dt id="flowtorch.bijector.Bijector.volume_preserving">
<code class="sig-name descname">volume_preserving</code><em class="property"> = True</em><a class="headerlink" href="#flowtorch.bijector.Bijector.volume_preserving" title="Permalink to this definition">¶</a></dt>
<dd></dd></dl>

</dd></dl>

</div>
<div class="section" id="module-flowtorch.param">
<span id="flowtorch-param-module"></span><h2>flowtorch.param module<a class="headerlink" href="#module-flowtorch.param" title="Permalink to this headline">¶</a></h2>
<dl class="py class">
<dt id="flowtorch.param.Params">
<em class="property">class </em><code class="sig-prename descclassname">flowtorch.param.</code><code class="sig-name descname">Params</code><a class="headerlink" href="#flowtorch.param.Params" title="Permalink to this definition">¶</a></dt>
<dd><p>Bases: <code class="xref py py-class docutils literal notranslate"><span class="pre">object</span></code></p>
<p>Deferred initialization of parameters.</p>
<dl class="py method">
<dt id="flowtorch.param.Params.build">
<code class="sig-name descname">build</code><span class="sig-paren">(</span><em class="sig-param"><span class="n">input_shape</span><span class="p">:</span> <span class="n">torch.Size</span></em>, <em class="sig-param"><span class="n">param_shapes</span><span class="p">:</span> <span class="n">Sequence<span class="p">[</span>torch.Size<span class="p">]</span></span></em><span class="sig-paren">)</span> &#x2192; Tuple<span class="p">[</span>torch.nn.modules.container.ModuleList<span class="p">, </span>Dict<span class="p">[</span>str<span class="p">, </span>torch.Tensor<span class="p">]</span><span class="p">]</span><a class="headerlink" href="#flowtorch.param.Params.build" title="Permalink to this definition">¶</a></dt>
<dd></dd></dl>

<dl class="py method">
<dt id="flowtorch.param.Params.forward">
<code class="sig-name descname">forward</code><span class="sig-paren">(</span><em class="sig-param"><span class="n">x</span><span class="p">:</span> <span class="n">torch.Tensor</span></em>, <em class="sig-param"><span class="n">modules</span><span class="p">:</span> <span class="n">Optional<span class="p">[</span>torch.nn.modules.container.ModuleList<span class="p">]</span></span> <span class="o">=</span> <span class="default_value">None</span></em><span class="sig-paren">)</span> &#x2192; Optional<span class="p">[</span>Sequence<span class="p">[</span>torch.Tensor<span class="p">]</span><span class="p">]</span><a class="headerlink" href="#flowtorch.param.Params.forward" title="Permalink to this definition">¶</a></dt>
<dd></dd></dl>

</dd></dl>

<dl class="py class">
<dt id="flowtorch.param.ParamsModule">
<em class="property">class </em><code class="sig-prename descclassname">flowtorch.param.</code><code class="sig-name descname">ParamsModule</code><span class="sig-paren">(</span><em class="sig-param"><span class="n">params</span><span class="p">:</span> <span class="n"><a class="reference internal" href="#flowtorch.param.Params" title="flowtorch.param.Params">flowtorch.param.Params</a></span></em>, <em class="sig-param"><span class="n">modules</span><span class="p">:</span> <span class="n">Optional<span class="p">[</span>torch.nn.modules.container.ModuleList<span class="p">]</span></span> <span class="o">=</span> <span class="default_value">None</span></em>, <em class="sig-param"><span class="n">buffers</span><span class="p">:</span> <span class="n">Optional<span class="p">[</span>Dict<span class="p">[</span>str<span class="p">, </span>torch.Tensor<span class="p">]</span><span class="p">]</span></span> <span class="o">=</span> <span class="default_value">None</span></em><span class="sig-paren">)</span><a class="headerlink" href="#flowtorch.param.ParamsModule" title="Permalink to this definition">¶</a></dt>
<dd><p>Bases: <code class="xref py py-class docutils literal notranslate"><span class="pre">torch.nn.modules.module.Module</span></code></p>
<dl class="py method">
<dt id="flowtorch.param.ParamsModule.forward">
<code class="sig-name descname">forward</code><span class="sig-paren">(</span><em class="sig-param"><span class="n">x</span><span class="p">:</span> <span class="n">torch.Tensor</span></em>, <em class="sig-param"><span class="n">context</span><span class="p">:</span> <span class="n">Optional<span class="p">[</span>torch.Tensor<span class="p">]</span></span> <span class="o">=</span> <span class="default_value">None</span></em><span class="sig-paren">)</span> &#x2192; Optional<span class="p">[</span>Sequence<span class="p">[</span>torch.Tensor<span class="p">]</span><span class="p">]</span><a class="headerlink" href="#flowtorch.param.ParamsModule.forward" title="Permalink to this definition">¶</a></dt>
<dd><p>Defines the computation performed at every call.</p>
<p>Should be overridden by all subclasses.</p>
<div class="admonition note">
<p class="admonition-title">Note</p>
<p>Although the recipe for forward pass needs to be defined within
this function, one should call the <code class="xref py py-class docutils literal notranslate"><span class="pre">Module</span></code> instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.</p>
</div>
</dd></dl>

<dl class="py attribute">
<dt id="flowtorch.param.ParamsModule.training">
<code class="sig-name descname">training</code><em class="property">: bool</em><a class="headerlink" href="#flowtorch.param.ParamsModule.training" title="Permalink to this definition">¶</a></dt>
<dd></dd></dl>

</dd></dl>

<dl class="py class">
<dt id="flowtorch.param.ParamsModuleList">
<em class="property">class </em><code class="sig-prename descclassname">flowtorch.param.</code><code class="sig-name descname">ParamsModuleList</code><span class="sig-paren">(</span><em class="sig-param"><span class="n">params_modules</span><span class="p">:</span> <span class="n">Sequence<span class="p">[</span><a class="reference internal" href="#flowtorch.param.ParamsModule" title="flowtorch.param.ParamsModule">ParamsModule</a><span class="p">]</span></span></em><span class="sig-paren">)</span><a class="headerlink" href="#flowtorch.param.ParamsModuleList" title="Permalink to this definition">¶</a></dt>
<dd><p>Bases: <code class="xref py py-class docutils literal notranslate"><span class="pre">torch.nn.modules.module.Module</span></code></p>
<dl class="py method">
<dt id="flowtorch.param.ParamsModuleList.forward">
<code class="sig-name descname">forward</code><span class="sig-paren">(</span><em class="sig-param"><span class="n">x</span><span class="p">:</span> <span class="n">torch.Tensor</span></em><span class="sig-paren">)</span> &#x2192; Sequence<span class="p">[</span>Optional<span class="p">[</span>Sequence<span class="p">[</span>torch.Tensor<span class="p">]</span><span class="p">]</span><span class="p">]</span><a class="headerlink" href="#flowtorch.param.ParamsModuleList.forward" title="Permalink to this definition">¶</a></dt>
<dd><p>Defines the computation performed at every call.</p>
<p>Should be overridden by all subclasses.</p>
<div class="admonition note">
<p class="admonition-title">Note</p>
<p>Although the recipe for forward pass needs to be defined within
this function, one should call the <code class="xref py py-class docutils literal notranslate"><span class="pre">Module</span></code> instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.</p>
</div>
</dd></dl>

<dl class="py attribute">
<dt id="flowtorch.param.ParamsModuleList.params_modules">
<code class="sig-name descname">params_modules</code><em class="property">: torch.nn.modules.container.ModuleList</em><a class="headerlink" href="#flowtorch.param.ParamsModuleList.params_modules" title="Permalink to this definition">¶</a></dt>
<dd></dd></dl>

</dd></dl>

</div>
<div class="section" id="module-flowtorch.utils">
<span id="flowtorch-utils-module"></span><h2>flowtorch.utils module<a class="headerlink" href="#module-flowtorch.utils" title="Permalink to this headline">¶</a></h2>
<dl class="py function">
<dt id="flowtorch.utils.clamp_preserve_gradients">
<code class="sig-prename descclassname">flowtorch.utils.</code><code class="sig-name descname">clamp_preserve_gradients</code><span class="sig-paren">(</span><em class="sig-param"><span class="n">x</span><span class="p">:</span> <span class="n">torch.Tensor</span></em>, <em class="sig-param"><span class="n">min</span><span class="p">:</span> <span class="n">float</span></em>, <em class="sig-param"><span class="n">max</span><span class="p">:</span> <span class="n">float</span></em><span class="sig-paren">)</span> &#x2192; torch.Tensor<a class="headerlink" href="#flowtorch.utils.clamp_preserve_gradients" title="Permalink to this definition">¶</a></dt>
<dd></dd></dl>

<dl class="py function">
<dt id="flowtorch.utils.clipped_sigmoid">
<code class="sig-prename descclassname">flowtorch.utils.</code><code class="sig-name descname">clipped_sigmoid</code><span class="sig-paren">(</span><em class="sig-param"><span class="n">x</span><span class="p">:</span> <span class="n">torch.Tensor</span></em><span class="sig-paren">)</span> &#x2192; torch.Tensor<a class="headerlink" href="#flowtorch.utils.clipped_sigmoid" title="Permalink to this definition">¶</a></dt>
<dd></dd></dl>

</div>
<div class="section" id="module-flowtorch">
<span id="module-contents"></span><h2>Module contents<a class="headerlink" href="#module-flowtorch" title="Permalink to this headline">¶</a></h2>
</div>
