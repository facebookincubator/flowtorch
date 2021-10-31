---
id: tests
title: Tests
sidebar_label: Tests
---

All `bijector.Bijector` and `params.Parameters` classes are covered by unit tests that test that the interface is satisfied, correct shape information is produced, and in the case of bijectors, that the log determinate absolute Jacobian is correct, amongst other things.

In general, you will not need to write new unit tests. When you implement a new component it will be detected by the library and included in existing tests.
