{{ fullname | escape | underline }}

.. automodule:: {{ fullname }}
   :members:
   :undoc-members:
   :show-inheritance:

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
