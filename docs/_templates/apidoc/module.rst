{{ fullname.split('.')[-1] | escape | underline }}

.. automodule:: {{ fullname }}
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

{% if subpackages %}
Subpackages
-----------

.. toctree::
   :maxdepth: 1

{% for item in subpackages %}
   {{ item }}
{% endfor %}
{% endif %}

{% if submodules %}
Submodules
----------

.. toctree::
   :maxdepth: 1

{% for item in submodules %}
   {{ item }}
{% endfor %}
{% endif %}
