---
id: tests
title: Tests
sidebar_label: Tests
---

All `bijector.Bijector` and `params.Params` classes are covered by unit tests that test that the interface is satisfied, correct shape information is produced, and in the case of bijectors, that the log determinate absolute Jacobian is correct.

In general, you will not need to write new unit but can hook into existing ones. You can examine how existing classes are tested and add your class to the same tests accordingly.

:::note
We will include more information on what tests are preformed and how to add unit tests for your contributed classes in the v2 version of the docs!
:::